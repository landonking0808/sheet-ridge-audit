import json, glob, sys, numpy as np
tag = sys.argv[1]
recs = [json.load(open(f)) for f in sorted(glob.glob(f'/tmp/vcdata/v03/dev/{tag}/*.json'))]
print('seg   area v02 -> v03   plateau v02 -> v03   n v02 -> v03  seams geom')
for r in recs:
    a, b = r['v02'], r['v03']
    print(f"{r['seg']:6s} {a['suspect_area_fraction']:.3f} -> {b['suspect_area_fraction']:.3f}   {a['winding_change_fraction']:.3f} -> {b['winding_change_fraction']:.3f}   {a['n_suspects']:5d} -> {b['n_suspects']:5d}  {b['n_winding_seams']:4d} {b['n_geom_steps']:3d}")
A = np.array([[r['v02']['suspect_area_fraction'], r['v03']['suspect_area_fraction']] for r in recs])
print('median area', np.median(A, 0).round(4), 'mean', A.mean(0).round(4), 'n segs', len(recs))
for ver in ('v02', 'v03'):
    items = [it for r in recs for it in r[ver]['items']]
    sus = [it for it in items if it['stratum'] != 'control']; ctl = [it for it in items if it['stratum'] == 'control']
    kept = [it for it in sus if it['at']]; drop = [it for it in sus if not it['at']]
    lab = lambda L: {k: sum(it['label'] == k for it in L) for k in ('REAL', 'PLAUSIBLE', 'FALSE')}
    print(ver, 'reviewed suspects', len(sus), 'kept', len(kept), lab(kept), 'dropped', len(drop), lab(drop))
    if kept:
        R = sum(it['label'] == 'REAL' for it in kept); RP = sum(it['label'] != 'FALSE' for it in kept)
        print(f'   precision among kept: REAL {R}/{len(kept)}={R/len(kept):.2f}  REAL+PL {RP}/{len(kept)}={RP/len(kept):.2f}')
    print('   controls', len(ctl), 'now flagged', sum(1 for it in ctl if it['at']), lab([it for it in ctl if it['at']]))
    # ranking: percentile rank of kept items within their segment
    for L in ('REAL', 'PLAUSIBLE', 'FALSE'):
        pr = [it['at'] / it['n_sus'] for it in kept if it['label'] == L]
        if pr: print(f'   {L:9s} rank pct median {np.median(pr):.3f}  (n={len(pr)})  top10%: {sum(p<=0.1 for p in pr)}')
    bykind = {}
    for it in kept:
        k = it['at_kinds']; bykind.setdefault(k, []).append(it['label'])
    for k, v in bykind.items():
        print(f'   kind {k:28s} n={len(v)} REAL={v.count("REAL")} PL={v.count("PLAUSIBLE")} FALSE={v.count("FALSE")}')
