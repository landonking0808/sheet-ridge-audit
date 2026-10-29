"""Scalar fields the auditor samples, and the backend interface.

A *field* is anything that can be sampled at level-0 volume coordinates of the
segment's frame (the frame the tifxyz coordinates are written in):

    class MyField(Field):
        name = "my-field"
        ridge_is_high = True      # high value = sheet centre (False: gap is high)
        voxel_size = 8.0          # level-0 voxels per field voxel (sampling step hint)
        typical_spacing = None    # sheet spacing in level-0 voxels, if known
        def sample(self, points_zyx):   # (N, 3) float level-0 voxels -> (N,) float
            ...                         # NaN where the field has no data

Optional:
    def prefetch(self, points_zyx): ...   # hint: these points come next
    def describe(self) -> dict: ...       # metadata for summary.json
    def close(self): ...

The raw-CT backend (:class:`ZarrField`) streams only the chunks the points
touch from a local or remote OME-Zarr (zarr v2), with an LRU chunk cache.
Other backends (e.g. Lasagna prediction fields) plug in through
:func:`sheet_ridge_audit.backends.load_backend`.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict

import numpy as np


class Field:
    name: str = "field"
    ridge_is_high: bool = True
    voxel_size: float = 1.0
    typical_spacing: float | None = None

    def sample(self, points_zyx: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def prefetch(self, points_zyx: np.ndarray) -> None:
        return None

    def describe(self) -> dict:
        return {"name": self.name, "ridge_is_high": bool(self.ridge_is_high),
                "voxel_size": float(self.voxel_size), "typical_spacing": self.typical_spacing}

    def close(self) -> None:
        return None


class _Adapter(Field):
    """Wraps a duck-typed field: anything with ``sample``; metadata taken from
    attributes or from a ``meta`` dict (keys ``name``, ``ridge_is_high`` or
    ``peak_is_sheet``, ``voxel_size`` or ``voxel_scale``, ``typical_spacing``)."""

    def __init__(self, obj):
        self.obj = obj
        meta = getattr(obj, "meta", None)
        meta = meta if isinstance(meta, dict) else {}

        def pick(*names, default=None):
            for n in names:
                if hasattr(obj, n) and not callable(getattr(obj, n)):
                    return getattr(obj, n)
                if n in meta and not callable(meta[n]):
                    return meta[n]
            return default
        self.name = str(pick("name", default=type(obj).__name__))
        self.ridge_is_high = bool(pick("ridge_is_high", "peak_is_sheet", default=True))
        self.voxel_size = float(pick("voxel_size", "voxel_scale", default=1.0) or 1.0)
        ts = pick("typical_spacing", default=None)
        self.typical_spacing = float(ts) if ts else None
        self._meta = {k: v for k, v in meta.items() if not callable(v)}

    def sample(self, points_zyx):
        return np.asarray(self.obj.sample(points_zyx), np.float32)

    def prefetch(self, points_zyx):
        f = getattr(self.obj, "prefetch", None)
        return f(points_zyx) if f else None

    def describe(self):
        d = super().describe()
        f = getattr(self.obj, "describe", None)
        if f:
            d.update(f())
        d["backend_meta"] = self._meta
        return d


def as_field(obj) -> Field:
    """Accept a :class:`Field` as is, or wrap any object with ``sample``."""
    if isinstance(obj, Field):
        return obj
    if not hasattr(obj, "sample"):
        raise TypeError(f"{obj!r} has no sample(points_zyx) method")
    return _Adapter(obj)


class ArrayField(Field):
    """An in-memory numpy volume (tests, synthetic data). Trilinear, NaN outside."""

    def __init__(self, vol: np.ndarray, name="array", ridge_is_high=True, scale=1.0, is_cos_field=False):
        self.vol = np.asarray(vol, np.float32)
        self.is_cos_field = bool(is_cos_field)
        self.name = name
        self.ridge_is_high = ridge_is_high
        self.voxel_size = float(scale)
        self.scale = float(scale)

    def sample(self, points_zyx):
        from scipy.ndimage import map_coordinates
        p = np.asarray(points_zyx, np.float64) / self.scale
        out = map_coordinates(self.vol, p.T, order=1, mode="constant", cval=np.nan)
        inside = np.all((p >= 0) & (p <= np.array(self.vol.shape) - 1), 1)
        out[~inside] = np.nan
        return out.astype(np.float32)


# --------------------------------------------------------------------------- zarr

def _open_mapper(url: str):
    import fsspec
    return fsspec.get_mapper(url)


def open_store(url: str):
    """(filesystem, root path) for a local path or any fsspec URL."""
    import fsspec
    fs, root = fsspec.core.url_to_fs(url.rstrip("/"))
    return fs, root.rstrip("/")


class ChunkCache:
    """Thread-safe LRU of decoded chunks, bounded in bytes."""

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self._d: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
                self.hits += 1
            return v

    def __contains__(self, key):
        with self._lock:
            return key in self._d

    def __len__(self):
        return len(self._d)

    @property
    def nbytes(self):
        return self._bytes

    def put(self, key, arr):
        with self._lock:
            if key in self._d:
                return
            self._d[key] = arr
            self._bytes += arr.nbytes
            self.misses += 1
            while self._bytes > self.max_bytes and len(self._d) > 1:
                _, old = self._d.popitem(last=False)
                self._bytes -= old.nbytes


class ZarrLevel:
    """One zarr v2 array (one OME level), read chunk by chunk.

    ``store`` is a URL/path string or an ``(fs, root)`` pair. Chunks are fetched
    concurrently with ``fs.cat``; a chunk that does not exist (404 / no file) is
    the fill value, or NaN with ``missing='nan'``; any other fetch error is
    retried and then raised (never silently turned into data)."""

    def __init__(self, store, path: str, cache: ChunkCache, missing: str = "fill", retries: int = 3):
        import numcodecs
        self.fs, self.root = open_store(store) if isinstance(store, str) else store
        self.path = path.strip("/")
        meta = json.loads(self._cat_one(self._k(".zarray")))
        if meta.get("zarr_format", 2) != 2:
            raise ValueError("only zarr v2 arrays are supported")
        self.order = meta.get("order", "C")
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill = meta.get("fill_value") or 0
        self.sep = meta.get("dimension_separator", ".")
        self.compressor = numcodecs.get_codec(meta["compressor"]) if meta.get("compressor") else None
        self.filters = [numcodecs.get_codec(f) for f in (meta.get("filters") or [])]
        self.cache = cache
        self.missing = missing
        self.retries = int(retries)
        self.nchunks = tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))
        self.bytes_fetched = 0
        self.chunks_fetched = 0
        self.chunks_missing = 0
        self._lock = threading.Lock()

    def _k(self, name):
        parts = [p for p in (self.root, self.path, name) if p]
        return "/".join(parts)

    def _cat_one(self, p):
        return self.fs.cat(p)

    def key(self, ci):
        return self._k(self.sep.join(str(int(i)) for i in ci))

    def _decode(self, buf):
        if self.compressor is not None:
            buf = self.compressor.decode(buf)
        for f in reversed(self.filters):
            buf = f.decode(buf)
        a = np.frombuffer(buf, dtype=self.dtype) if not isinstance(buf, np.ndarray) else buf.view(self.dtype).ravel()
        return a.reshape(self.chunks, order=self.order)

    def _fetch(self, keys):
        """{key: bytes | None (absent)}; retries transient errors."""
        import time as _t
        out = {}
        todo = list(keys)
        for attempt in range(self.retries + 1):
            if not todo:
                break
            try:
                if len(todo) == 1:
                    got = {todo[0]: self._cat_safe(todo[0])}
                elif getattr(self.fs, "async_impl", False):
                    got = self.fs.cat(todo, on_error="return", batch_size=min(len(todo), 512))
                else:
                    got = self.fs.cat(todo, on_error="return")
            except Exception as e:  # whole-batch failure
                got = {k: e for k in todo}
            again = []
            for k in todo:
                v = got.get(k)
                if isinstance(v, (bytes, bytearray, memoryview)):
                    out[k] = bytes(v)
                elif v is None or isinstance(v, (FileNotFoundError, KeyError)):
                    out[k] = None
                else:
                    again.append(k)
            todo = again
            if todo and attempt < self.retries:
                _t.sleep(0.5 * 2 ** attempt)
        if todo:
            raise IOError(f"chunk fetch failed after {self.retries} retries: {todo[0]} ({len(todo)} chunks)")
        return out

    def _cat_safe(self, k):
        try:
            return self.fs.cat(k)
        except Exception as e:  # noqa: BLE001 - classified by the caller
            return e

    def ensure(self, chunk_ids, batch=1024):
        """Fetch (concurrently) every chunk in ``chunk_ids`` not yet cached."""
        need = []
        seen = set()
        for ci in chunk_ids:
            t = tuple(int(v) for v in ci)
            if t in seen or (self.path, t) in self.cache:
                continue
            seen.add(t)
            need.append(t)
        for i in range(0, len(need), batch):
            part = need[i:i + batch]
            keys = [self.key(ci) for ci in part]
            got = self._fetch(keys)
            for ci, k in zip(part, keys):
                buf = got.get(k)
                if buf is None:
                    with self._lock:
                        self.chunks_missing += 1
                    arr = _Fill(np.nan if self.missing == "nan" else self.fill, self.chunks, self.dtype)
                else:
                    with self._lock:
                        self.bytes_fetched += len(buf)
                        self.chunks_fetched += 1
                    arr = self._decode(buf)
                self.cache.put((self.path, ci), arr)

    def get_chunk(self, ci):
        v = self.cache.get((self.path, ci))
        if v is None:
            self.ensure([ci])
            v = self.cache.get((self.path, ci))
        return v

    def chunk_ids_for(self, idx_float):
        """Chunks touched by trilinear interpolation at float indices (N, 3)."""
        q = np.floor(np.asarray(idx_float, np.float64))
        q = q[np.isfinite(q).all(1)].astype(np.int64)
        if not len(q):
            return np.zeros((0, 3), np.int64)
        cs = np.array(self.chunks)
        nc = np.array(self.nchunks, np.int64)
        a = q // cs
        spill = ((q + 1) // cs != a).any(1)   # corner in the next chunk (few points)
        ids = [a]
        if spill.any():
            qa, qb = a[spill], (q[spill] + 1) // cs
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        if dz or dy or dx:
                            ids.append(np.stack([qb[:, 0] if dz else qa[:, 0], qb[:, 1] if dy else qa[:, 1],
                                                 qb[:, 2] if dx else qa[:, 2]], 1))
        ids = np.concatenate(ids)
        ids = ids[np.all((ids >= 0) & (ids < nc), 1)]
        if not len(ids):
            return ids
        flat = np.unique((ids[:, 0] * nc[1] + ids[:, 1]) * nc[2] + ids[:, 2])  # 1-D unique: fast
        return np.stack([flat // (nc[1] * nc[2]), (flat // nc[2]) % nc[1], flat % nc[2]], 1)

    def lookup(self, ijk: np.ndarray, out_of_bounds=np.nan) -> np.ndarray:
        """Values at integer voxel indices (M, 3); NaN outside the array."""
        ijk = np.asarray(ijk, np.int64)
        out = np.full(len(ijk), out_of_bounds, np.float32)
        inside = np.all((ijk >= 0) & (ijk < np.array(self.shape)), 1)
        if not inside.any():
            return out
        idx = np.nonzero(inside)[0]
        q = ijk[idx]
        cs = np.array(self.chunks)
        ci = q // cs
        local = q - ci * cs
        nc = np.array(self.nchunks, np.int64)
        flat = (ci[:, 0] * nc[1] + ci[:, 1]) * nc[2] + ci[:, 2]
        uniq, inv = np.unique(flat, return_inverse=True)
        cids = np.stack([uniq // (nc[1] * nc[2]), (uniq // nc[2]) % nc[1], uniq % nc[2]], 1)
        self.ensure(cids)
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
        for u in range(len(uniq)):
            sel = order[bounds[u]:bounds[u + 1]]
            ch = self.get_chunk(tuple(int(v) for v in cids[u]))
            l = local[sel]
            out[idx[sel]] = ch[l[:, 0], l[:, 1], l[:, 2]]
        return out

    def interp(self, idx_float, max_block_voxels=4_000_000, decode=None):
        """Trilinear interpolation at float array indices (N, 3), NaN outside
        [0, shape-1] or where a corner is a NaN (missing) chunk.

        Fast path: the touched chunks are pasted into one dense block covering
        the points' bounding box and interpolated with
        ``scipy.ndimage.map_coordinates``; large or scattered point sets are
        split into octants first. ``decode`` maps raw values to physical ones
        (applied to the block, so trilinear runs on decoded values)."""
        p = np.asarray(idx_float, np.float64).reshape(-1, 3)
        out = np.full(len(p), np.nan, np.float32)
        hi = np.asarray(self.shape, np.float64) - 1
        ok = np.isfinite(p).all(1)
        ok[ok] = np.all((p[ok] >= 0) & (p[ok] <= hi), 1)
        if ok.any():
            sel = np.nonzero(ok)[0]
            cids = self.chunk_ids_for(p[sel])
            chunk_bytes = float(np.prod(self.chunks)) * self.dtype.itemsize
            if len(cids) * chunk_bytes < 0.5 * self.cache.max_bytes:
                self.ensure(cids)  # one concurrent fetch for all of them
            out[sel] = self._interp_rec(p[sel], max_block_voxels, decode, 0)
        return out

    def _interp_rec(self, p, maxv, decode, depth):
        from scipy.ndimage import map_coordinates
        lo = np.floor(p.min(0)).astype(np.int64)
        hi = np.minimum(np.floor(p.max(0)).astype(np.int64) + 1, np.asarray(self.shape) - 1)
        ext = hi - lo + 1
        if np.prod(ext.astype(np.float64)) > maxv and len(p) > 1 and depth < 12:
            ax = int(np.argmax(ext))
            mid = np.median(p[:, ax])
            left = p[:, ax] <= mid
            if left.all() or (~left).all():
                left = np.arange(len(p)) < len(p) // 2
            res = np.empty(len(p), np.float32)
            res[left] = self._interp_rec(p[left], maxv, decode, depth + 1)
            res[~left] = self._interp_rec(p[~left], maxv, decode, depth + 1)
            return res
        cids = self.chunk_ids_for(p)
        self.ensure(cids)
        cs = np.asarray(self.chunks)
        chunks = [(ci, self.get_chunk(tuple(int(v) for v in ci))) for ci in cids]
        nan_fill = any(isinstance(ch, _Fill) and not np.isfinite(ch.fill) for _, ch in chunks)
        float_path = decode is not None or nan_fill
        block = np.zeros(tuple(int(e) for e in ext), np.float32 if float_path else self.dtype)
        shape = np.asarray(self.shape)
        for ci, ch in chunks:
            c0 = ci * cs
            c1 = np.minimum(c0 + cs, shape)
            a = np.maximum(c0, lo)
            b = np.minimum(c1, hi + 1)
            if np.any(b <= a):
                continue
            dst = tuple(slice(int(x - y), int(z - y)) for x, z, y in zip(a, b, lo))
            if isinstance(ch, _Fill):
                block[dst] = ch.fill
            else:
                block[dst] = ch[tuple(slice(int(x - y), int(z - y)) for x, z, y in zip(a, b, c0))]
        if decode is not None:
            block = np.asarray(decode(block), np.float32)
        q = (p - lo).T
        if float_path:
            nan = np.isnan(block)
            if nan.any():  # NaN-aware: a NaN corner gives NaN
                v = map_coordinates(np.where(nan, 0, block), q, order=1, mode="nearest", prefilter=False,
                                    output=np.float32)
                m = map_coordinates(nan.astype(np.float32), q, order=1, mode="nearest", prefilter=False,
                                    output=np.float32)
                v[m > 1e-6] = np.nan
                return v
        return map_coordinates(block, q, order=1, mode="nearest", prefilter=False, output=np.float32)


class _Fill:
    """Stand-in for a missing chunk: indexes to the fill value."""

    def __init__(self, fill, shape, dtype):
        self.fill, self.shape, self.dtype = fill, shape, dtype
        self.nbytes = 64

    def __getitem__(self, idx):
        n = len(idx[0]) if isinstance(idx, tuple) else 1
        return np.full(n, self.fill, np.float32)


def trilinear(lookup, p: np.ndarray, shape=None) -> np.ndarray:
    """Trilinear interpolation of ``lookup(int_ijk)`` at float points (N, 3).
    Points outside [0, shape-1] get NaN (same convention as
    ``scipy.ndimage.map_coordinates(order=1)`` inside the array)."""
    p = np.asarray(p, np.float64)
    out = np.full(len(p), np.nan, np.float32)
    fin = np.isfinite(p).all(1)
    if shape is not None:
        hi = np.asarray(shape, np.float64) - 1
        fin &= np.all((p >= 0) & (p <= hi), 1) if fin.any() else fin
    if not fin.any():
        return out
    q = p[fin]
    f = np.floor(q)
    if shape is not None:
        f = np.minimum(f, np.maximum(np.asarray(shape) - 2, 0))
    t = (q - f).astype(np.float32)
    f = f.astype(np.int64)
    acc = np.zeros(len(q), np.float32)
    corners = []
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                corners.append(f + np.array([dz, dy, dx]))
    vals = lookup(np.concatenate(corners, 0)).reshape(8, len(q))
    i = 0
    for dz in (0, 1):
        wz = t[:, 0] if dz else 1 - t[:, 0]
        for dy in (0, 1):
            wy = t[:, 1] if dy else 1 - t[:, 1]
            for dx in (0, 1):
                wx = t[:, 2] if dx else 1 - t[:, 2]
                acc += wz * wy * wx * vals[i]
                i += 1
    out[fin] = acc
    return out


class ZarrField(Field):
    """Raw CT from an OME-Zarr (or plain zarr v2 array), sampled trilinearly.

    ``url`` may be an OME-Zarr group (``level`` selects ``<url>/<level>`` and
    its multiscales scale/translation) or a single array. Coordinates passed to
    :meth:`sample` are level-0 voxels; level-k index = (p - translation) / scale.
    """

    ridge_is_high = True

    def __init__(self, url: str, level: int = 0, cache_mb: int = 768, name: str | None = None):
        self.url = url.rstrip("/")
        self.level = int(level)
        self.store = open_store(self.url)
        fs, root = self.store
        self.cache = ChunkCache(cache_mb * 2**20)
        self.scale = np.ones(3)
        self.translation = np.zeros(3)
        path = ""
        try:
            attrs = json.loads(fs.cat(f"{root}/.zattrs"))
        except Exception:
            attrs = {}
        ms = attrs.get("multiscales")
        if ms:
            ds = ms[0]["datasets"]
            entry = ds[self.level]
            path = entry["path"]
            base = None
            for t in entry.get("coordinateTransformations", []):
                if t.get("type") == "scale":
                    self.scale = np.array(t["scale"][-3:], float)
                if t.get("type") == "translation":
                    self.translation = np.array(t["translation"][-3:], float)
            for t in ds[0].get("coordinateTransformations", []):
                if t.get("type") == "scale":
                    base = np.array(t["scale"][-3:], float)
            if base is not None:  # express relative to level 0
                self.scale = self.scale / base
                self.translation = self.translation / base
        else:
            try:
                json.loads(fs.cat(f"{root}/.zarray"))
            except Exception:
                path = str(self.level)
                self.scale = np.full(3, 2.0 ** self.level)
        self.arr = ZarrLevel(self.store, path, self.cache)
        self.shape_l0 = tuple(int(s * k) for s, k in zip(self.arr.shape, self.scale))
        self.name = name or f"ct:L{self.level}"
        self.voxel_size = float(self.scale.mean())

    def to_level(self, points_zyx):
        return (np.asarray(points_zyx, np.float64) - self.translation) / self.scale

    def chunk_ids(self, points_zyx):
        return self.arr.chunk_ids_for(self.to_level(points_zyx))

    def prefetch(self, points_zyx):
        self.arr.ensure(self.chunk_ids(points_zyx))

    def sample(self, points_zyx):
        return self.arr.interp(self.to_level(points_zyx))

    def describe(self):
        d = super().describe()
        d.update(url=self.url, level=self.level, scale_zyx=self.scale.tolist(),
                 translation_zyx=self.translation.tolist(), level_shape_zyx=list(self.arr.shape),
                 chunks=list(self.arr.chunks), dtype=str(self.arr.dtype),
                 chunks_fetched=self.arr.chunks_fetched, mb_fetched=round(self.arr.bytes_fetched / 1e6, 1),
                 cache_hits=self.cache.hits)
        return d
