"""PNG maps and CT cross-sections."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from .tifxyz import grid_normals, grid_tangents  # noqa: E402


def _aspect_fig(h, w, base=9.0):
    ar = h / max(w, 1)
    W = base
    H = float(np.clip(base * ar, 2.5, 16)) + 0.8
    return W, H


def _extent(grid):
    r, c = grid.orig_rows, grid.orig_cols
    hs = 0.5 * grid.stride / grid.upsample
    return [c[0] - hs, c[-1] + hs, r[-1] + hs, r[0] - hs]


def map_png(path, grid, img, title, cmap="viridis", vmin=None, vmax=None, cbar_label=""):
    h, w = img.shape
    fig, ax = plt.subplots(figsize=_aspect_fig(h, w))
    a = np.ma.masked_invalid(img)
    cm = plt.get_cmap(cmap).with_extremes(bad="#202020")
    im = ax.imshow(a, cmap=cm, vmin=vmin, vmax=vmax, extent=_extent(grid), interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("tifxyz grid col")
    ax.set_ylabel("tifxyz grid row")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _norm_img(x, valid):
    v = x[valid & np.isfinite(x)]
    if not len(v):
        return np.zeros_like(x)
    lo, hi = np.percentile(v, [1, 99])
    out = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    out[~(valid & np.isfinite(x))] = np.nan
    return out


def _outline(ax, X, Y, mask, color, lw):
    if mask.sum() >= 1:
        ax.contour(X, Y, ndi.binary_dilation(mask).astype(float), levels=[0.5], colors=[color], linewidths=lw)


def overview_png(path, grid, m, suspects, labels, title, top=40):
    """Two panels over the original tifxyz grid: c0 (field at the mesh) and
    the unwrapped winding offset, each with the suspects outlined and numbered."""
    h, w = m.valid.shape
    ext = _extent(grid)
    wide = w > 1.6 * h
    ar = h / max(w, 1)
    if wide:
        W = 15.0
        fig, axes = plt.subplots(2, 1, figsize=(W, float(np.clip(2 * W * ar, 4, 18)) + 1.6))
    else:
        H = 8.5
        fig, axes = plt.subplots(1, 2, figsize=(float(np.clip(2 * H / max(ar, 1e-3), 8, 20)) + 1.5, H + 1.2))
    X = grid.orig_cols[None, :].repeat(h, 0)
    Y = grid.orig_rows[:, None].repeat(w, 1)
    panels = [
        (m.c0_smooth, "RdBu", -1, 1, "c0: field at the mesh (smoothed); blue = on a ridge, red = in a gap"),
        (np.where(m.trusted, m.rel_offset, np.nan), "PuOr_r", -1.5, 1.5,
         "winding offset (sheets) vs the dominant sheet; |x|>0.5 = other sheet"),
    ]
    fs = 9 if len(suspects) > 25 else 10
    for ax, (img, cmap, lo, hi, ttl) in zip(np.atleast_1d(axes), panels):
        cm = plt.get_cmap(cmap).with_extremes(bad="#3a3a3a")
        im = ax.imshow(np.ma.masked_invalid(img), cmap=cm, vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        for s in suspects[:top]:
            reg = labels == s.id
            col = "#000000" if "winding" not in s.kinds else "#00a000"
            _outline(ax, X, Y, reg, col, 1.3)
        for s in suspects[:top]:
            ax.annotate(str(s.id), (s.col, s.row), color="black", fontsize=fs, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffff66", alpha=0.85, linewidth=0.5))
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("tifxyz grid col", fontsize=9)
        ax.set_ylabel("tifxyz grid row", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#000000", lw=1.5, label="suspect: off ridge (gap)"),
               Line2D([0], [0], color="#00a000", lw=1.5, label="suspect: on another sheet (winding)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _plane_trace(seg, p0, a_dir, n_dir, half_w, half_h, r, c):
    """Points where the mesh crosses the plane through p0 spanned by a_dir, n_dir.
    Returns (a, h, row, col) arrays of crossing points in plane coords."""
    bn = np.cross(a_dir, n_dir)
    bn /= np.linalg.norm(bn) + 1e-9
    H, W = seg.valid.shape
    reach = int(np.ceil(2.5 * max(half_w, half_h) / max(seg.node_spacing * 0.5, 1e-6))) + 2
    r0, r1 = max(0, r - reach), min(H, r + reach + 1)
    c0, c1 = max(0, c - reach), min(W, c + reach + 1)
    P = seg.zyx[r0:r1, c0:c1] - p0
    ok = seg.valid[r0:r1, c0:c1]
    d = P @ bn
    out = []
    for dr, dc in ((0, 1), (1, 0)):
        A, B = P[: P.shape[0] - dr, : P.shape[1] - dc], P[dr:, dc:]
        da, db = d[: d.shape[0] - dr, : d.shape[1] - dc], d[dr:, dc:]
        e = ok[: ok.shape[0] - dr, : ok.shape[1] - dc] & ok[dr:, dc:] & (np.sign(da) != np.sign(db))
        if not e.any():
            continue
        t = (da[e] / (da[e] - db[e]))[:, None]
        X = A[e] + t * (B[e] - A[e])
        ii, jj = np.nonzero(e)
        out.append(np.stack([X @ a_dir, X @ n_dir, ii + r0, jj + c0], 1))
    if not out:
        return np.zeros((0, 4))
    o = np.concatenate(out)
    keep = (np.abs(o[:, 0]) <= half_w) & (np.abs(o[:, 1]) <= half_h)
    return o[keep]


def cross_section_png(path, seg, field, suspect, spacing, label_mask=None, grid=None,
                      half_w=None, half_h=None, res=None, title="", overlay=None):
    """Two slices through the suspect centre, each orthogonal to the surface
    (one along the grid's column direction, one along its row direction), from
    ``field`` (raw CT, normally). The mesh's intersection with each slice is
    drawn (cyan; magenta inside the suspect). ``overlay`` (e.g. the Lasagna cos
    field) adds its ridge lines (cos = 0.7 contours, red)."""
    r, c = suspect.seg_rc
    nrm = grid_normals(seg.zyx, seg.valid)
    tu, tv = grid_tangents(seg.zyx, seg.valid)
    p0 = seg.zyx[r, c].astype(np.float64)
    n = nrm[r, c].astype(np.float64)
    if not np.isfinite(n).all():
        return False
    b = suspect.bbox_grid
    node = seg.node_spacing * seg.upsample
    ext_vox = node * max(b[2] - b[0], b[3] - b[1], 1.0)
    half_w = half_w or float(np.clip(0.6 * ext_vox + 3 * spacing, 4 * spacing, 8 * spacing))
    half_h = half_h or float(np.clip(2.5 * spacing, 12, 300))
    vs = getattr(field, "voxel_size", 1.0) or 1.0
    res = res or max(0.5, vs, half_w / 350)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, tdir, nm in ((axes[0], tu[r, c], "slice along grid cols (u)"), (axes[1], tv[r, c], "slice along grid rows (v)")):
        t = np.asarray(tdir, np.float64)
        if not np.isfinite(t).all():
            ax.set_axis_off()
            continue
        t = t - (t @ n) * n
        t /= np.linalg.norm(t) + 1e-9
        A = np.arange(-half_w, half_w + 1e-6, res)
        Hh = np.arange(-half_h, half_h + 1e-6, res)
        aa, hh = np.meshgrid(A, Hh)
        pts = (p0 + aa[..., None] * t + hh[..., None] * n).reshape(-1, 3)
        if hasattr(field, "prefetch"):
            field.prefetch(pts)
        img = np.asarray(field.sample(pts), np.float32).reshape(aa.shape)
        ok = np.isfinite(img)
        lo, hi = np.percentile(img[ok], [1, 99.5]) if ok.any() else (0, 1)
        ax.imshow(np.ma.masked_invalid(img), cmap="gray", vmin=lo, vmax=hi, origin="lower",
                  extent=[A[0], A[-1], Hh[0], Hh[-1]], interpolation="bilinear")
        if overlay is not None:
            ores = max(res, 0.5 * (getattr(overlay, "voxel_size", 1.0) or 1.0))
            A2 = np.arange(-half_w, half_w + 1e-6, ores)
            H2 = np.arange(-half_h, half_h + 1e-6, ores)
            a2, h2 = np.meshgrid(A2, H2)
            ov = np.asarray(overlay.sample((p0 + a2[..., None] * t + h2[..., None] * n).reshape(-1, 3)),
                            np.float32).reshape(a2.shape)
            if np.isfinite(ov).any():
                ax.contour(A2, H2, np.nan_to_num(ov, nan=-1), levels=[0.7], colors=["#ff3030"], linewidths=0.7,
                           alpha=0.8)
        tr = _plane_trace(seg, p0, t, n, half_w, half_h, r, c)
        if len(tr):
            inside = np.zeros(len(tr), bool)
            if label_mask is not None and grid is not None:
                ri = np.clip(np.searchsorted(grid.rows, tr[:, 2]), 0, len(grid.rows) - 1)
                ci = np.clip(np.searchsorted(grid.cols, tr[:, 3]), 0, len(grid.cols) - 1)
                inside = label_mask[ri, ci]
            ax.scatter(tr[~inside, 0], tr[~inside, 1], s=5, c="#00e0ff", linewidths=0, label="mesh")
            ax.scatter(tr[inside, 0], tr[inside, 1], s=6, c="#ff30ff", linewidths=0, label="mesh in suspect")
        if overlay is not None:
            ax.plot([], [], color="#ff3030", lw=1, label="Lasagna ridge (cos 0.7)")
        ax.plot([0], [0], "+", color="#ffff00", ms=12)
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel("voxels along the surface")
        ax.set_ylabel("voxels along the mesh normal")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7, markerscale=3, framealpha=0.6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return True
