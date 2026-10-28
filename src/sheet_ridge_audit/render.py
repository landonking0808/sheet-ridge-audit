"""PNG maps and CT cross-sections."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from .tifxyz import grid_normals, grid_tangents  # noqa: E402


def _aspect_fig(h, w, base=9.0):
    ar = h / max(w, 1)
    W = base
    H = float(np.clip(base * ar, 2.5, 16)) + 0.8
    return W, H


def _extent(grid):
    r, c = grid.rows, grid.cols
    return [c[0] - 0.5 * grid.stride, c[-1] + 0.5 * grid.stride, r[-1] + 0.5 * grid.stride, r[0] - 0.5 * grid.stride]


def map_png(path, grid, img, title, cmap="viridis", vmin=None, vmax=None, cbar_label=""):
    h, w = img.shape
    fig, ax = plt.subplots(figsize=_aspect_fig(h, w))
    a = np.ma.masked_invalid(img)
    cm = plt.get_cmap(cmap).with_extremes(bad="#202020")
    im = ax.imshow(a, cmap=cm, vmin=vmin, vmax=vmax, extent=_extent(grid), interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
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


def suspects_png(path, grid, metrics, suspects, labels, title, top=30):
    img = _norm_img(metrics.center_value, metrics.valid)
    h, w = img.shape
    fig, ax = plt.subplots(figsize=_aspect_fig(h, w))
    cm = plt.get_cmap("gray").with_extremes(bad="#301818")
    ext = _extent(grid)
    ax.imshow(np.ma.masked_invalid(img), cmap=cm, extent=ext, interpolation="nearest")
    colors = {"off_sheet": "#ffb000", "step": "#ff2050", "drift": "#40c0ff", "winding": "#ff40ff", "off_ridge": "#40c0ff", "residue": "#ffffff"}
    X = grid.cols[None, :].repeat(h, 0)
    Y = grid.rows[:, None].repeat(w, 1)
    # outlines of the kind masks (thin), then suspect outlines (thick)
    for k in ("off_sheet", "off_ridge", "winding"):
        mk = metrics.masks.get(k)
        if mk is not None and mk.any():
            ax.contour(X, Y, mk.astype(float), levels=[0.5], colors=[colors[k]], linewidths=0.6, alpha=0.8)
    st = metrics.masks.get("step")
    if st is not None and st.any():
        rr, cc = np.nonzero(st)
        ax.scatter(grid.cols[cc], grid.rows[rr], s=2, c=colors["step"], linewidths=0)
    for s in suspects[:top]:
        reg = labels == labels[s.sub_rc]
        if reg.sum() >= 2:
            ax.contour(X, Y, ndi.binary_dilation(reg).astype(float), levels=[0.5], colors=["#00ff60"], linewidths=1.2)
        ax.text(s.col, s.row, str(s.id), color="#00ff60", fontsize=8, ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.5, pad=0.5, linewidth=0))
    handles = [Patch(color=colors[k], label=k) for k in ("off_sheet", "off_ridge", "step", "winding")]
    handles.append(Patch(color="#00ff60", label="suspect (ranked)"))
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.6)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
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
                      half_w=None, half_h=None, res=None, title=""):
    """Two CT slices through the suspect centre, each orthogonal to the surface
    (one along the grid's column direction, one along its row direction), with
    the mesh trace drawn in colour (magenta inside the suspect region)."""
    r, c = suspect.row, suspect.col
    nrm = grid_normals(seg.zyx, seg.valid)
    tu, tv = grid_tangents(seg.zyx, seg.valid)
    p0 = seg.zyx[r, c].astype(np.float64)
    n = nrm[r, c].astype(np.float64)
    if not np.isfinite(n).all():
        return False
    b = suspect.bbox_grid
    ext_vox = seg.node_spacing * max(b[2] - b[0], b[3] - b[1], 1)
    half_w = half_w or float(np.clip(0.75 * ext_vox + 4 * spacing, 6 * spacing, 400))
    half_h = half_h or float(np.clip(max(3.0 * spacing, 0.3 * half_w), 12, 150))
    vs = getattr(field, "voxel_size", 1.0) or 1.0
    res = res or max(0.5, min(vs, half_w / 300))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, tdir, nm in ((axes[0], tu[r, c], "along grid cols (u)"), (axes[1], tv[r, c], "along grid rows (v)")):
        t = np.asarray(tdir, np.float64)
        if not np.isfinite(t).all():
            ax.set_axis_off()
            continue
        t = t - (t @ n) * n
        t /= np.linalg.norm(t) + 1e-9
        A = np.arange(-half_w, half_w + 1e-6, res)
        Hh = np.arange(-half_h, half_h + 1e-6, res)
        aa, hh = np.meshgrid(A, Hh)
        pts = p0 + aa[..., None] * t + hh[..., None] * n
        if hasattr(field, "prefetch"):
            field.prefetch(pts.reshape(-1, 3))
        img = np.asarray(field.sample(pts.reshape(-1, 3)), np.float32).reshape(aa.shape)
        ok = np.isfinite(img)
        if ok.any():
            lo, hi = np.percentile(img[ok], [1, 99.5])
        else:
            lo, hi = 0, 1
        ax.imshow(np.ma.masked_invalid(img), cmap="gray", vmin=lo, vmax=hi, origin="lower",
                  extent=[A[0], A[-1], Hh[0], Hh[-1]], interpolation="bilinear")
        tr = _plane_trace(seg, p0, t, n, half_w, half_h, r, c)
        if len(tr):
            inside = np.zeros(len(tr), bool)
            if label_mask is not None and grid is not None:
                ri = np.clip(np.searchsorted(grid.rows, tr[:, 2]), 0, len(grid.rows) - 1)
                ci = np.clip(np.searchsorted(grid.cols, tr[:, 3]), 0, len(grid.cols) - 1)
                inside = label_mask[ri, ci]
            ax.scatter(tr[~inside, 0], tr[~inside, 1], s=4, c="#00e0ff", linewidths=0, label="mesh")
            ax.scatter(tr[inside, 0], tr[inside, 1], s=5, c="#ff30ff", linewidths=0, label="mesh in suspect")
        ax.axhline(0, color="#ffff00", lw=0.4, alpha=0.4)
        for k in (-1, 1):
            ax.axhline(k * spacing, color="#ffff00", lw=0.4, alpha=0.25, ls="--")
        ax.plot([0], [0], "+", color="#ffff00", ms=10)
        ax.set_title(f"{nm}", fontsize=9)
        ax.set_xlabel("voxels along surface")
        ax.set_ylabel("voxels along normal")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7, markerscale=3, framealpha=0.5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True
