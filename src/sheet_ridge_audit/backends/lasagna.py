"""Lasagna prediction-field backend (built in).

Samples the public **Lasagna** predictions of the Vesuvius Challenge
(``villa/lasagna``, ScrollPrize/villa, predict3d output; model
``20260419180421``), published on
``s3://vesuvius-challenge-open-data/<sample>/representations/predictions/lasagna/``,
at points given in the **full-resolution scan frame** (z, y, x level-0 voxels of
the scan whose id prefixes the Lasagna folder, ``base_shape_zyx`` in the
manifest). Lasagna is the work of the villa/lasagna authors; this module only
reads its published outputs. The encodings below were read from villa ``main`` @
``322cb0899ac9`` and checked on five real segments (see the README).

Encodings:

* ``cos``: ``u8 = clip(c01*255)``, ``c01 = 0.5 + 0.5*cos(2*pi*winding)``
  (``train_unet_3d.compute_targets_3d``; ``tifxyz_labels``: 1 on a labelled sheet
  surface, 0 midway between sheets). Returned as the physical cosine
  ``2*u8/255 - 1`` in [-1, 1] (``decode='phys'``, default) or ``u8/255``
  (``decode='unit'``). **+1 = on a sheet ridge.**
* ``grad_mag``: ``u8 = clip(g*1000)``, ``g`` in windings per inference-input
  voxel; windings per full-res voxel = ``u8 / (grad_mag_encode_scale /
  grad_mag_factor)``. Raw 0 = no prediction (NaN here). On real data the
  measured cos period is about 1.12-1.34 x ``1/grad_mag`` (grad_mag overstates
  the sheet density), so the auditor measures the period from the cos
  profiles and uses grad_mag only for its *local variation*.
* ``nx``, ``ny``: ``u8 = round(n*s*127 + 128)`` with ``s = sign(nz)``: an
  unsigned (axial) unit normal. :meth:`LasagnaField.normal` interpolates the
  outer product ``n n^T`` so the sign flip at ``nz = 0`` does no harm.
* Frame: array index = ``p_fullres / 2**level`` (no half-voxel shift), as in
  Lasagna's ``grid_sample_3d_u8`` kernel. ``scaledown`` in the manifest is the
  OME level.

Chunks are fetched lazily and concurrently (fsspec) into an in-memory LRU;
nothing is written to disk. Absent chunks (no prediction) read as NaN.
"""
from __future__ import annotations

import json
import posixpath
import re

import numpy as np

from ..fields import ChunkCache, Field, ZarrLevel, open_store

S3_HTTP = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
DEFAULT_PERIOD_FACTOR = 1.2   # measured cos period / (1/grad_mag), median of 5 real segments (1.12-1.34)

# sample -> (scan id, voxel size um, manifest key, raw CT volume folder). Checked 2026-10-28.
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
    """Raw CT OME-Zarr of the scan the Lasagna field (and ``-on-<scan>`` meshes) live on."""
    return f"{S3_HTTP}{sample}/volumes/{REGISTRY[sample][3]}"


def sample_for(src: str) -> str | None:
    """Registry name for a registry key, manifest URL, run folder or scan id."""
    if src in REGISTRY:
        return src
    for k, v in REGISTRY.items():
        if v[2] in src or posixpath.dirname(v[2]) in src.rstrip("/"):
            return k
    for k, v in REGISTRY.items():
        if v[0] in src:
            return k
    return None


def sample_for_segment(path: str) -> str | None:
    """Guess the registry sample from a segment path ``...-on-<scan id>-...tifxyz``."""
    m = re.search(r"-on-(\d{14})", path)
    if not m:
        return None
    for k, v in REGISTRY.items():
        if v[0] == m.group(1):
            return k
    return None


class LasagnaField(Field):
    """One channel of a published Lasagna prediction, sampled in the full-res scan frame.

    LasagnaField("PHerc0139")                                 # registry name
    LasagnaField("https://.../X.lasagna.json", field="grad_mag")
    LasagnaField("https://.../<run folder>/")                  # the one *.lasagna.json in it
    LasagnaField(local_manifest_path, field="cos", level=4)    # coarser level of the same pyramid
    """

    FIELDS = ("cos", "grad_mag", "nx", "ny")
    ridge_is_high = True
    typical_spacing = None

    def __init__(self, manifest_or_base_url: str, field: str = "cos", level: int | None = None,
                 cache_mb: int = 256, decode: str = "phys", cache_bytes: int | None = None):
        if field not in self.FIELDS:
            raise ValueError(f"field must be one of {self.FIELDS}")
        src = str(manifest_or_base_url)
        sample = src if src in REGISTRY else None
        if src in REGISTRY:
            src = manifest_url(src)
        if not src.endswith(".lasagna.json"):
            src = self._find_manifest(src)
        self.manifest_url = src
        fs, mpath = open_store(src)
        self.manifest = json.loads(fs.cat(mpath))
        base_dir = src.rsplit("/", 1)[0]
        groups = self.manifest["groups"]
        grp = next((g for g in groups.values() if field in g.get("channels", [])), None)
        if grp is None:
            raise ValueError(f"manifest has no channel {field!r}")
        zpath = grp["zarr"].rstrip("/")
        native = int(grp["scaledown"])
        omeroot, _, lvl = zpath.rpartition("/")
        if not lvl.isdigit() or int(lvl) != native:
            omeroot = zpath  # legacy layout: the path is the array itself
            if level not in (None, native):
                raise ValueError("this manifest has no OME pyramid; only the native level is available")
            arr_url = f"{base_dir}/{zpath}"
        else:
            arr_url = f"{base_dir}/{omeroot}/{native if level is None else int(level)}"
        self.level = native if level is None else int(level)
        self.native_level = native
        self.field = field
        self.decode_mode = decode
        s2b = float(self.manifest.get("source_to_base", 1.0))
        self.scale = (2 ** self.level) * s2b      # full-res voxels per field voxel
        self.voxel_size = float(self.scale)
        nbytes = cache_bytes if cache_bytes is not None else int(cache_mb) << 20
        self.cache = ChunkCache(nbytes)
        self.arr = ZarrLevel(arr_url, "", self.cache, missing="nan")
        self.array_url = arr_url
        self.grad_mag_decode_scale = (float(self.manifest.get("grad_mag_encode_scale", 1000.0))
                                      / float(self.manifest.get("grad_mag_factor", 1.0)))
        base = self.manifest.get("base_shape_zyx")
        self.shape_l0 = tuple(int(v) for v in base) if base else None
        if sample is None:
            sample = sample_for(src)
        self.sample_name = sample
        self.scan_id = REGISTRY[sample][0] if sample else None
        vox_um = REGISTRY[sample][1] if sample else None
        self.fullres_voxel_um = vox_um
        stem = posixpath.basename(src)[: -len(".lasagna.json")]
        self.name = f"lasagna:{stem}:{field}"
        self.meta = {
            "name": self.name, "sample": sample, "scan_id": self.scan_id, "field": field,
            "level": self.level, "native_level": native, "array_url": arr_url,
            "array_shape_zyx": list(self.arr.shape), "base_shape_zyx": list(base) if base else None,
            "voxel_scale": self.scale, "fullres_voxel_um": vox_um,
            "field_voxel_um": (vox_um * self.scale) if vox_um else None,
            "grad_mag_decode_scale": self.grad_mag_decode_scale,
            "peak_is_sheet": True,
            "units": {"cos": "cos(2*pi*winding) in [-1,1]" if decode == "phys" else "0.5+0.5cos in [0,1]",
                      "grad_mag": "windings per full-res voxel", "nx": "unit", "ny": "unit"}[field],
            "index_convention": "idx = p_fullres / voxel_scale (lasagna grid_sample_3d_u8)",
            "credit": "Lasagna predictions, ScrollPrize/villa lasagna (model 20260419180421)",
        }
        self._partners = {}

    @staticmethod
    def _find_manifest(base: str) -> str:
        base = base.rstrip("/")
        if base.startswith("http") and ".s3.amazonaws.com/" in base:
            import urllib.parse
            import urllib.request
            root, _, prefix = base.partition(".s3.amazonaws.com/")
            q = urllib.parse.urlencode({"list-type": "2", "prefix": prefix + "/", "delimiter": "/"})
            x = urllib.request.urlopen(f"{root}.s3.amazonaws.com/?{q}", timeout=60).read().decode()
            keys = [k for k in re.findall(r"<Key>([^<]*)</Key>", x) if k.endswith(".lasagna.json")]
            if len(keys) != 1:
                raise ValueError(f"expected one .lasagna.json under {base}, found {keys}")
            return f"{root}.s3.amazonaws.com/{keys[0]}"
        fs, root = open_store(base)
        found = [p for p in fs.ls(root, detail=False) if str(p).endswith(".lasagna.json")]
        if len(found) != 1:
            raise ValueError(f"expected one .lasagna.json under {base}, found {found}")
        p = str(found[0])
        if base.startswith(("http://", "https://")):
            return base + "/" + posixpath.basename(p)
        if "://" in base:
            return base.split("://", 1)[0] + "://" + p
        return p

    @property
    def is_cos_field(self) -> bool:
        """Values are cos(2 pi winding) in [-1, 1] (the auditor then uses p(0) as c0)."""
        return self.field == "cos" and self.decode_mode == "phys"

    # ------------------------------------------------------------------ decoding
    def decode(self, raw):
        raw = np.asarray(raw, np.float32)
        if self.field == "cos":
            return raw / 255.0 if self.decode_mode == "unit" else raw * (2.0 / 255.0) - 1.0
        if self.field == "grad_mag":
            return np.where(raw == 0, np.nan, raw / self.grad_mag_decode_scale)
        return (raw - 128.0) / 127.0

    def to_index(self, points_zyx):
        return np.asarray(points_zyx, np.float64).reshape(-1, 3) / self.scale

    def prefetch(self, points_zyx):
        self.arr.ensure(self.arr.chunk_ids_for(self.to_index(points_zyx)))

    def sample(self, points_zyx):
        """Decoded trilinear values at full-res points (N, 3) zyx; NaN outside the
        array, in absent chunks, and (grad_mag) where the raw value is 0."""
        if self.field == "grad_mag":   # raw 0 = missing: decode (mask) before interpolating
            return self.arr.interp(self.to_index(points_zyx), decode=self.decode)
        return self.decode(self.arr.interp(self.to_index(points_zyx))).astype(np.float32)  # affine decode

    def sample_raw(self, points_zyx):
        return self.arr.interp(self.to_index(points_zyx))

    # ------------------------------------------------------------------ helpers
    def partner(self, field: str) -> "LasagnaField":
        """Another channel of the same manifest (opened once, at its native level)."""
        if field == self.field:
            return self
        if field not in self._partners:
            self._partners[field] = LasagnaField(self.manifest_url, field=field,
                                                 cache_bytes=max(32 << 20, self.cache.max_bytes // 4))
        return self._partners[field]

    def sheet_spacing_from_grad_mag(self, points_zyx):
        """1 / grad_mag in full-res voxels (raw; the measured cos period is ~1.2x this)."""
        g = self.partner("grad_mag").sample(points_zyx)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(g > 0, 1.0 / g, np.nan).astype(np.float32)

    def local_period(self, points_zyx):
        """Prior for the local sheet period (voxels): DEFAULT_PERIOD_FACTOR / grad_mag.
        The auditor rescales it by the period it measures from the cos profiles."""
        return DEFAULT_PERIOD_FACTOR * self.sheet_spacing_from_grad_mag(points_zyx)

    def prefetch_local_period(self, points_zyx):
        self.partner("grad_mag").prefetch(points_zyx)

    def normal(self, points_zyx):
        """Unit sheet normal (N, 3) zyx, sign-free (returned with n_z >= 0), from nx/ny;
        interpolated through the outer product n n^T of the 8 corners."""
        fx, fy = self.partner("nx"), self.partner("ny")
        p = np.asarray(points_zyx, np.float64).reshape(-1, 3) / fx.scale
        out = np.full((len(p), 3), np.nan)
        shp = np.array(fx.arr.shape)
        ok = np.all(np.isfinite(p), 1) & np.all((p >= 0) & (p <= shp - 1), 1)
        if not ok.any():
            return out
        q = p[ok]
        f = np.minimum(np.floor(q).astype(np.int64), shp - 2).clip(0)
        t = q - f
        corners = [(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
        allidx = np.concatenate([f + np.array(c) for c in corners])
        rx = fx.arr.lookup(allidx).reshape(8, -1).astype(np.float64)
        ry = fy.arr.lookup(allidx).reshape(8, -1).astype(np.float64)
        valid = np.isfinite(rx) & np.isfinite(ry) & ~((rx == 0) & (ry == 0))
        nx = np.nan_to_num((rx - 128.0) / 127.0)
        ny = np.nan_to_num((ry - 128.0) / 127.0)
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, None))
        w = np.stack([(t[:, 0] if dz else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
                      * (t[:, 2] if dx else 1 - t[:, 2]) for dz, dy, dx in corners]) * valid
        v = np.stack([nz, ny, nx], -1)
        v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        T = np.einsum("km,kmi,kmj->mij", w, v, v)
        _, vecs = np.linalg.eigh(T)
        e = vecs[:, :, -1]
        e *= np.where(e[:, 0] < 0, -1.0, 1.0)[:, None]
        e[w.sum(0) < 0.5] = np.nan
        out[ok] = e
        return out

    def ct_volume_url(self) -> str | None:
        """Raw CT of the same scan (registry), for cross-sections."""
        return volume_url(self.sample_name) if self.sample_name else None

    def describe(self):
        d = super().describe()
        d.update({k: v for k, v in self.meta.items()})
        d.update(manifest=self.manifest_url, chunks_fetched=self.arr.chunks_fetched,
                 chunks_missing=self.arr.chunks_missing, mb_fetched=round(self.arr.bytes_fetched / 1e6, 2))
        gm = self._partners.get("grad_mag")
        if gm is not None:
            d["grad_mag_mb_fetched"] = round(gm.arr.bytes_fetched / 1e6, 2)
        return d


def make_field(volume, level=None, **opts):
    """Backend factory: ``volume`` = registry name (e.g. ``PHerc0139``), manifest
    URL/path, or run folder."""
    if not volume:
        raise ValueError("lasagna backend needs --scroll NAME or --lasagna MANIFEST_URL")
    return LasagnaField(volume, level=level, **opts)


def registry_table() -> str:
    lines = [f"{'sample':<12} {'scan':<15} {'um':>6}  manifest"]
    for k, v in REGISTRY.items():
        lines.append(f"{k:<12} {v[0]:<15} {v[1]:>6}  {S3_HTTP}{v[2]}")
    return "\n".join(lines)


__all__ = ["LasagnaField", "REGISTRY", "manifest_url", "volume_url", "sample_for", "sample_for_segment",
           "make_field", "DEFAULT_PERIOD_FACTOR", "registry_table"]
