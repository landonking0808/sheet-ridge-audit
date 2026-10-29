"""Lasagna field backend for sheet-ridge-audit.

Samples the public Lasagna predictions (villa/lasagna predict3d output, published
on s3://vesuvius-challenge-open-data/<sample>/representations/predictions/lasagna/)
at arbitrary points given in the FULL-RESOLUTION scan frame (z, y, x voxels of the
scan whose id prefixes the Lasagna folder, i.e. `base_shape_zyx` in the manifest).

Encodings (villa lasagna @ 322cb08):
  cos      : u8 = clip(c01*255) where c01 = 0.5 + 0.5*cos(2*pi*winding)
             (train_unet_3d.compute_targets_3d; tifxyz_labels: cos_full = 1 on a
             labelled sheet surface, 0 midway between sheets). The fitter decodes
             t/255 (fit_data.py). We return the physical cosine 2*t/255 - 1 in [-1, 1]
             (decode='phys', default) or t/255 (decode='unit').
  grad_mag : u8 = clip(g*1000) with g in windings per *inference-input* voxel;
             manifest grad_mag_factor = 1/input_sd converts to base voxels:
             windings/base-voxel = t / (grad_mag_encode_scale / grad_mag_factor)
             (fit_data.load: gmag_enc = encode_scale / grad_mag_factor).
  nx, ny   : u8 = round(n*s*127 + 128), s = sign(nz) so nz >= 0 (normal_encoding.py);
             decode (t-128)/127, nz = sqrt(1 - nx^2 - ny^2). Unsigned (axial) normal.
  Frame    : array index = fullres_coord / 2**level (no half-voxel shift), the
             convention of lasagna's grid_sample_3d_u8 kernel ((p - origin)*inv_scale,
             origin 0 for whole-volume OME output).

Only numpy + numcodecs + fsspec. Chunks are fetched lazily (concurrently, via
fsspec `cat`) and kept in an in-memory LRU. Nothing is written to disk.
"""
from __future__ import annotations

import json
import math
import posixpath
from collections import OrderedDict
from typing import Iterable

import numpy as np

S3_HTTP = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"

# sample -> (scan id, voxel size um, manifest key, raw volume long id). Verified 2026-10-28.
REGISTRY = {
    "PHerc0125": ("20250821151825", 9.362, "PHerc0125/representations/predictions/lasagna/20250821151825-lasagna-20260419180421/PHerc0125.lasagna.json", "20250821151825-9.362um-1.2m-113keV-masked.zarr"),
    "PHerc0139": ("20260102150214", 2.399, "PHerc0139/representations/predictions/lasagna/20260102150214-lasagna-20260419180421-L2/PHerc0139-20260102150214-lasagna-20260724.lasagna.json", "20260102150214-2.399um-0.2m-78keV-masked.zarr"),
    "PHerc0191": ("20250821151635", 9.362, "PHerc0191/representations/predictions/lasagna/20250821151635-lasagna-20260419180421/PHerc0191.lasagna.json", "20250821151635-9.362um-1.2m-113keV-masked.zarr"),
    "PHerc0211": ("20250821151803", 9.362, "PHerc0211/representations/predictions/lasagna/20250821151803-lasagna-20260419180421/PHerc0211.lasagna.json", "20250821151803-9.362um-1.2m-113keV-masked.zarr"),
    "PHerc0257": ("20250821151750", 9.362, "PHerc0257/representations/predictions/lasagna/20250821151750-lasagna-20260419180421/PHerc0257.lasagna.json", "20250821151750-9.362um-1.2m-113keV-masked.zarr"),
    "PHerc0268": ("20251110183117", 8.64, "PHerc0268/representations/predictions/lasagna/20251110183117-lasagna-20260419180421/PHerc0268.lasagna.json", "20251110183117-8.640um-1.2m-116keV-masked.zarr"),
    "PHerc0332": ("20251211183505", 2.399, "PHerc0332/representations/predictions/lasagna/20251211183505-lasagna-20260419180421-L2/PHerc0332-20251211183505-lasagna-20260724.lasagna.json", "20251211183505-2.399um-0.2m-78keV-masked.zarr"),
    "PHerc0343": ("20250521140437", 8.64, "PHerc0343/representations/predictions/lasagna/20250521140437-lasagna-20260419180421/PHerc0343.lasagna.json", "20250521140437-8.640um-1.2m-116keV-masked.zarr"),
    "PHerc0358": ("20250821151737", 9.362, "PHerc0358/representations/predictions/lasagna/20250821151737-lasagna-20260419180421/PHerc0358.lasagna.json", "20250821151737-9.362um-1.2m-113keV-masked.zarr"),
    "PHerc0800": ("20250521135224", 8.64, "PHerc0800/representations/predictions/lasagna/20250521135224-lasagna-20260419180421/PHerc0800.lasagna.json", "20250521135224-8.640um-1.2m-116keV-masked.zarr"),
    "PHerc0813": ("20250821151723", 9.362, "PHerc0813/representations/predictions/lasagna/20250821151723-lasagna-20260419180421/PHerc0813.lasagna.json", "20250821151723-9.362um-1.2m-113keV-masked.zarr"),
    "PHerc0814": ("20260309142202", 2.399, "PHerc0814/representations/predictions/lasagna/20260309142202-lasagna-20260419180421-L2/PHerc0814-20260309142202-lasagna-20260724.lasagna.json", "20260309142202-2.399um-0.2m-78keV-masked.zarr"),
    "PHerc0826": ("20250821151701", 9.362, "PHerc0826/representations/predictions/lasagna/20250821151701-lasagna-20260419180421/PHerc0826.lasagna.json", "20250821151701-9.362um-1.2m-113keV-masked.zarr"),
    "PHerc1203": ("20250820131727", 9.362, "PHerc1203/representations/predictions/lasagna/20250820131727-lasagna-20260419180421/PHerc1203.lasagna.json", "20250820131727-9.362um-1.2m-113keV-masked.zarr"),
    "PHerc1218": ("20250521120456", 8.64, "PHerc1218/representations/predictions/lasagna/20250521120456-lasagna-20260419180421/PHerc1218.lasagna.json", "20250521120456-8.640um-1.2m-116keV-masked.zarr"),
    "PHerc1299": ("20260309130042", 2.399, "PHerc1299/representations/predictions/lasagna/20260309130042-lasagna-20260419180421-L2/PHerc1299-20260309130042-lasagna-20260724.lasagna.json", "20260309130042-2.399um-0.2m-78keV-masked.zarr"),
    "PHerc1447": ("20250521151220", 8.64, "PHerc1447/representations/predictions/lasagna/20250521151220-lasagna-20260419180421/PHerc1447.lasagna.json", "20250521151220-8.640um-1.2m-116keV-masked.zarr"),
    "PHerc1451": ("20260319101107", 2.399, "PHerc1451/representations/predictions/lasagna/20260319101107-lasagna-20260419180421-L2/PHerc1451-20260319101107-lasagna-20260724.lasagna.json", "20260319101107-2.399um-0.2m-78keV-masked.zarr"),
    "PHerc1545": ("20250821151648", 9.362, "PHerc1545/representations/predictions/lasagna/20250821151648-lasagna-20260419180421/PHerc1545.lasagna.json", "20250821151648-9.362um-1.2m-113keV-masked.zarr"),
    "PHerc1667": ("20251217075048", 2.399, "PHerc1667/representations/predictions/lasagna/20251217075048-lasagna-20260419180421-L2/PHerc1667-20251217075048-lasagna-20260724.lasagna.json", "20251217075048-2.399um-0.2m-78keV-masked.zarr"),
    "PHercParis3": ("20260427095331", 2.4, "PHercParis3/representations/predictions/lasagna/20260427095331-lasagna-20260419180421-L2/PHercParis3-20260427095331-lasagna-20260724.lasagna.json", "20260427095331-2.400um-0.2m-78keV-masked.zarr"),
    "PHercParis4": ("20260411134726", 2.4, "PHercParis4/representations/predictions/lasagna/20260411134726-lasagna-20260419180421-L2/PHercParis4-20260411134726-las-sd2-5b17ff6c.lasagna.json", "20260411134726-2.400um-0.2m-78keV-masked.zarr"),
}


def manifest_url(sample: str) -> str:
    return S3_HTTP + REGISTRY[sample][2]


def volume_url(sample: str) -> str:
    """Raw CT OME-Zarr of the scan the Lasagna field (and `-on-<scan>` meshes) live on."""
    return f"{S3_HTTP}{sample}/volumes/{REGISTRY[sample][3]}"


def _fs_for(url: str):
    import fsspec

    if "://" in url and not url.startswith("file://"):
        proto = url.split("://", 1)[0]
        return fsspec.filesystem(proto)
    return fsspec.filesystem("file")


class ZarrLevel:
    """Lazy, chunk-cached reader of one zarr v2 array (local path or http(s)/s3 URL)."""

    def __init__(self, url: str, cache_bytes: int = 512 << 20, fs=None):
        self.url = url.rstrip("/")
        self.fs = fs or _fs_for(self.url)
        za = json.loads(self.fs.cat(self.url + "/.zarray"))
        if za.get("zarr_format") != 2:
            raise ValueError("only zarr v2 arrays are supported")
        self.shape = tuple(int(v) for v in za["shape"])
        self.chunks = tuple(int(v) for v in za["chunks"])
        self.dtype = np.dtype(za["dtype"])
        self.fill = za.get("fill_value") or 0
        self.order = za.get("order", "C")
        self.sep = za.get("dimension_separator", ".")
        if za.get("filters"):
            raise ValueError("zarr filters not supported")
        comp = za.get("compressor")
        if comp:
            import numcodecs

            self.codec = numcodecs.get_codec(comp)
        else:
            self.codec = None
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_bytes = 0
        self.cache_limit = int(cache_bytes)
        self.n_fetched = 0
        self.bytes_fetched = 0

    def _key(self, c: tuple) -> str:
        return self.url + "/" + self.sep.join(str(i) for i in c)

    def _decode(self, raw: bytes) -> np.ndarray:
        buf = self.codec.decode(raw) if self.codec is not None else raw
        a = np.frombuffer(buf, dtype=self.dtype)
        return a.reshape(self.chunks, order=self.order)

    def _put(self, c, arr):
        self._cache[c] = arr
        self._cache_bytes += arr.nbytes
        while self._cache_bytes > self.cache_limit and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes

    def ensure(self, chunk_ids: Iterable[tuple]) -> None:
        missing = [c for c in set(map(tuple, chunk_ids)) if c not in self._cache]
        # Fetch in batches so the LRU never has to hold more than one batch above its limit.
        step = 256
        for i in range(0, len(missing), step):
            batch = missing[i:i + step]
            urls = [self._key(c) for c in batch]
            got = self.fs.cat(urls, on_error="return") if len(urls) > 1 else {urls[0]: self._cat1(urls[0])}
            for c, u in zip(batch, urls):
                raw = got.get(u)
                if isinstance(raw, (bytes, bytearray)):
                    arr = self._decode(raw)
                    self.n_fetched += 1
                    self.bytes_fetched += len(raw)
                elif raw is None or isinstance(raw, (FileNotFoundError, KeyError)) or "404" in str(raw) or "Not Found" in str(raw):
                    arr = np.full(self.chunks, self.fill, dtype=self.dtype)  # absent chunk = fill
                else:
                    raise IOError(f"chunk fetch failed {u}: {raw!r}")
                self._put(c, arr)

    def _cat1(self, u):
        try:
            return self.fs.cat(u)
        except FileNotFoundError as e:
            return e

    def chunk(self, c: tuple) -> np.ndarray:
        arr = self._cache.get(c)
        if arr is None:
            self.ensure([c])
            arr = self._cache[c]
        else:
            self._cache.move_to_end(c)
        return arr

    def gather(self, idx: np.ndarray) -> np.ndarray:
        """Values at integer voxel indices idx (N,3) zyx; out-of-bounds -> fill. Returns raw dtype."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.full(len(idx), self.fill, dtype=self.dtype)
        shp = np.array(self.shape)
        inb = np.all((idx >= 0) & (idx < shp), axis=1)
        if not inb.any():
            return out
        ii = idx[inb]
        cs = np.array(self.chunks)
        cid = ii // cs
        loc = ii - cid * cs
        # group by chunk
        key = np.ravel_multi_index(cid.T, tuple(int(math.ceil(s / c)) for s, c in zip(self.shape, self.chunks)))
        order = np.argsort(key, kind="stable")
        ukeys, starts = np.unique(key[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        chunk_list = [tuple(int(v) for v in cid[order[s]]) for s in starts]
        self.ensure(chunk_list)
        vals = np.empty(len(ii), dtype=self.dtype)
        for c, s, e in zip(chunk_list, starts, ends):
            sel = order[s:e]
            a = self.chunk(c)
            l = loc[sel]
            vals[sel] = a[l[:, 0], l[:, 1], l[:, 2]]
        out[inb] = vals
        return out

    def trilinear(self, pidx: np.ndarray, decode=None, mask_zero: bool = False) -> np.ndarray:
        """Trilinear interpolation at fractional array indices pidx (N,3) zyx.

        decode: applied to the 8 corner values (float) before blending.
        mask_zero: treat raw value 0 as missing (renormalise weights; NaN if < half the weight present).
        Points outside the array -> NaN.
        """
        p = np.asarray(pidx, dtype=np.float64)
        n = len(p)
        res = np.full(n, np.nan)
        shp = np.array(self.shape)
        ok = np.all(np.isfinite(p), axis=1) & np.all((p >= 0) & (p <= shp - 1), axis=1)
        if not ok.any():
            return res
        q = p[ok]
        f = np.floor(q).astype(np.int64)
        f = np.minimum(f, shp - 2).clip(0)
        t = q - f
        corners = [(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
        allidx = np.concatenate([f + np.array(c) for c in corners])
        raw = self.gather(allidx).reshape(8, -1).astype(np.float64)
        w = np.stack([
            (t[:, 0] if dz else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dx else 1 - t[:, 2])
            for dz, dy, dx in corners
        ])
        v = decode(raw) if decode is not None else raw
        if mask_zero:
            m = (raw != 0).astype(np.float64)
            ws = (w * m).sum(0)
            val = np.where(ws >= 0.5, (w * m * v).sum(0) / np.maximum(ws, 1e-12), np.nan)
        else:
            val = (w * v).sum(0)
        res[ok] = val
        return res


class LasagnaField:
    """One channel of a published Lasagna prediction, sampled in the full-res scan frame.

    LasagnaField("PHerc1667")                                # registry short name
    LasagnaField("https://.../X.lasagna.json", field="grad_mag")
    LasagnaField("https://.../<run folder>/")                 # folder: the one *.lasagna.json in it
    LasagnaField(local_manifest_path, field="cos", level=4)   # coarser OME level of the same pyramid
    """

    FIELDS = ("cos", "grad_mag", "nx", "ny")

    def __init__(self, manifest_or_base_url: str, field: str = "cos", level: int | None = None,
                 cache_bytes: int = 256 << 20, decode: str = "phys"):
        if field not in self.FIELDS:
            raise ValueError(f"field must be one of {self.FIELDS}")
        src = manifest_or_base_url
        sample = None
        if src in REGISTRY:
            sample = src
            src = manifest_url(src)
        self.fs = _fs_for(src)
        if not src.endswith(".lasagna.json"):
            src = self._find_manifest(src)
        self.manifest_url = src
        self.manifest = json.loads(self.fs.cat(src))
        self.base_dir = posixpath.dirname(src) if "://" in src else __import__("os").path.dirname(src)
        groups = self.manifest["groups"]
        grp = next((g for g in groups.values() if field in g.get("channels", [])), None)
        if grp is None:
            raise ValueError(f"manifest has no channel {field!r}")
        zpath = grp["zarr"]
        native_level = int(grp["scaledown"])
        omeroot, _, lvl = zpath.rstrip("/").rpartition("/")
        if not lvl.isdigit() or int(lvl) != native_level:
            omeroot, lvl = zpath, str(native_level)  # legacy layout: path is the array itself
            if level not in (None, native_level):
                raise ValueError("this manifest has no OME pyramid; only the native level is available")
        self.level = native_level if level is None else int(level)
        arr_url = f"{self.base_dir}/{omeroot}/{self.level}" if level is not None else f"{self.base_dir}/{zpath}"
        self.field = field
        self.decode_mode = decode
        s2b = float(self.manifest.get("source_to_base", 1.0))
        self.scale = (2 ** self.level) * s2b  # full-res voxels per array voxel
        self.arr = ZarrLevel(arr_url, cache_bytes=cache_bytes, fs=self.fs)
        gm_scale = float(self.manifest.get("grad_mag_encode_scale", 1000.0)) / float(self.manifest.get("grad_mag_factor", 1.0))
        self.grad_mag_decode_scale = gm_scale
        base = self.manifest.get("base_shape_zyx")
        if sample is None:
            sample = next((k for k, v in REGISTRY.items() if v[2] in src), None)
        scan_id = REGISTRY[sample][0] if sample else None
        vox_um = REGISTRY[sample][1] if sample else None
        self.meta = {
            "name": f"{posixpath.basename(src).removesuffix('.lasagna.json')}:{field}",
            "sample": sample,
            "scan_id": scan_id,
            "field": field,
            "level": self.level,
            "native_level": native_level,
            "array_url": arr_url,
            "array_shape_zyx": self.arr.shape,
            "base_shape_zyx": tuple(base) if base else None,
            "voxel_scale": self.scale,                # full-res voxels per field voxel
            "fullres_voxel_um": vox_um,
            "field_voxel_um": (vox_um * self.scale) if vox_um else None,
            "grad_mag_decode_scale": gm_scale,         # u8 / this = windings per full-res voxel
            "grad_mag_max_encodable": 255.0 / gm_scale,
            "peak_is_sheet": True,                     # cos = +1 on traced sheet surfaces (training target)
            "units": {"cos": "cos(2*pi*winding) in [-1,1]" if decode == "phys" else "0.5+0.5cos in [0,1]",
                      "grad_mag": "windings per full-res voxel", "nx": "unit", "ny": "unit"}[field],
            "index_convention": "idx = p_fullres / voxel_scale (lasagna grid_sample_3d_u8)",
        }
        self.meta["sheet_spacing_vox"] = self.sheet_spacing_from_grad_mag

    def _find_manifest(self, base: str) -> str:
        base = base.rstrip("/")
        if base.startswith("http") and "s3.amazonaws.com" in base:
            import re
            import urllib.parse
            import urllib.request

            root, _, prefix = base.partition(".s3.amazonaws.com/")
            q = urllib.parse.urlencode({"list-type": "2", "prefix": prefix + "/", "delimiter": "/"})
            x = urllib.request.urlopen(f"{root}.s3.amazonaws.com/?{q}").read().decode()
            keys = [k for k in re.findall(r"<Key>([^<]*)</Key>", x) if k.endswith(".lasagna.json")]
            if len(keys) != 1:
                raise ValueError(f"expected one .lasagna.json under {base}, found {keys}")
            return f"{root}.s3.amazonaws.com/{keys[0]}"
        found = [p for p in self.fs.ls(base, detail=False) if str(p).endswith(".lasagna.json")]
        if len(found) != 1:
            raise ValueError(f"expected one .lasagna.json under {base}, found {found}")
        p = str(found[0])
        return p if "://" in p or not base.startswith("file://") else "file://" + p

    # --- decoding -------------------------------------------------------------
    def decode(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        if self.field == "cos":
            return raw / 255.0 if self.decode_mode == "unit" else raw * (2.0 / 255.0) - 1.0
        if self.field == "grad_mag":
            return raw / self.grad_mag_decode_scale
        return (raw - 128.0) / 127.0

    def to_index(self, points_zyx: np.ndarray) -> np.ndarray:
        return np.asarray(points_zyx, dtype=np.float64) / self.scale

    def sample(self, points_zyx: np.ndarray) -> np.ndarray:
        """Decoded, trilinearly interpolated values at full-res points (N,3) zyx. NaN outside.

        grad_mag: raw 0 (no prediction / masked) is treated as missing.
        """
        pts = np.asarray(points_zyx, dtype=np.float64).reshape(-1, 3)
        return self.arr.trilinear(self.to_index(pts), decode=self.decode,
                                  mask_zero=(self.field == "grad_mag"))

    def sample_raw(self, points_zyx: np.ndarray) -> np.ndarray:
        return self.arr.trilinear(self.to_index(np.asarray(points_zyx, float).reshape(-1, 3)))

    # --- helpers ---------------------------------------------------------------
    def sheet_spacing_from_grad_mag(self, points_zyx: np.ndarray) -> np.ndarray:
        """Sheet spacing in full-res voxels = 1 / grad_mag (needs field='grad_mag')."""
        if self.field != "grad_mag":
            raise ValueError("sheet spacing needs a grad_mag field: LasagnaField(src, field='grad_mag')")
        g = self.sample(points_zyx)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(g > 0, 1.0 / g, np.nan)

    def normal(self, points_zyx: np.ndarray) -> np.ndarray:
        """Unit sheet normal (N,3) in zyx order, sign-free (axial; returned with n_z >= 0).

        Needs a field opened on 'nx' or 'ny' (the partner channel is opened from the same
        manifest). Interpolation is done on the outer product n n^T of the 8 decoded
        corner normals (sign-invariant), taking the principal eigenvector.
        """
        if self.field not in ("nx", "ny"):
            raise ValueError("normal() needs LasagnaField(src, field='nx' or 'ny')")
        other = getattr(self, "_partner", None)
        if other is None:
            other = LasagnaField(self.manifest_url, field="ny" if self.field == "nx" else "nx",
                                 level=self.level if self.level != self.meta["native_level"] else None)
            self._partner = other
        fx, fy = (self, other) if self.field == "nx" else (other, self)
        p = self.to_index(np.asarray(points_zyx, float).reshape(-1, 3))
        n = len(p)
        out = np.full((n, 3), np.nan)
        shp = np.array(self.arr.shape)
        ok = np.all(np.isfinite(p), axis=1) & np.all((p >= 0) & (p <= shp - 1), axis=1)
        if not ok.any():
            return out
        q = p[ok]
        f = np.minimum(np.floor(q).astype(np.int64), shp - 2).clip(0)
        t = q - f
        corners = [(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
        allidx = np.concatenate([f + np.array(c) for c in corners])
        rx = fx.arr.gather(allidx).reshape(8, -1).astype(np.float64)
        ry = fy.arr.gather(allidx).reshape(8, -1).astype(np.float64)
        nx = (rx - 128.0) / 127.0
        ny = (ry - 128.0) / 127.0
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, None))
        valid = ~((rx == 0) & (ry == 0))  # 0/0 = never written (fill)
        w = np.stack([
            (t[:, 0] if dz else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dx else 1 - t[:, 2])
            for dz, dy, dx in corners
        ]) * valid
        v = np.stack([nz, ny, nx], axis=-1)  # (8, M, 3) zyx
        v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        T = np.einsum("km,kmi,kmj->mij", w, v, v)
        _, vecs = np.linalg.eigh(T)
        e = vecs[:, :, -1]
        e *= np.where(e[:, 0] < 0, -1.0, 1.0)[:, None]
        e[w.sum(0) < 0.5] = np.nan
        out[ok] = e
        return out


class RawVolume:
    """Raw CT OME-Zarr level sampled in the full-res frame (index = p / 2**level, same as LasagnaField)."""

    def __init__(self, url: str, level: int = 0, cache_bytes: int = 768 << 20):
        self.level = int(level)
        self.scale = 2 ** self.level
        self.arr = ZarrLevel(url.rstrip("/") + f"/{self.level}", cache_bytes=cache_bytes)

    def sample(self, points_zyx: np.ndarray) -> np.ndarray:
        return self.arr.trilinear(np.asarray(points_zyx, float).reshape(-1, 3) / self.scale)
