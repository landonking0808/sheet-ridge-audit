import pickle, numpy as np
data = pickle.load(open('rank_data.pkl', 'rb'))
def score(f, opt):
    ext, d, c, dw, g, area, mc0 = f['ext'], f['depth'], f['conf'], f['dw'], f['geom'], f['area'], f['mc0']
    if opt == 'S1 v0.3 draft: ext*(depth+|dW|+step)*conf': return ext * (d + dw + (1 if g else 0)) * c
    if opt == 'S2 ext*depth*conf*(1+0.5|dW|)': return ext * max(d, 0.05) * c * (1 + 0.5 * dw + (0.5 if g else 0))
    if opt == 'S3 ext*depth*conf': return ext * max(d, 0.05) * c
    if opt == 'S4 area*(depth+|dW|) (v0.2-like)': return area * (d + dw)
for opt in ['S1 v0.3 draft: ext*(depth+|dW|+step)*conf', 'S2 ext*depth*conf*(1+0.5|dW|)', 'S3 ext*depth*conf', 'S4 area*(depth+|dW|) (v0.2-like)']:
    pr = {'REAL': [], 'PLAUSIBLE': [], 'FALSE': []}
    for sname, F, items in data:
        sc = np.array([score(f, opt) for f in F])
        order = np.argsort(-sc, kind='stable'); rank = np.empty(len(sc), int); rank[order] = np.arange(1, len(sc) + 1)
        for at, lab in items:
            pr[lab].append(rank[at - 1] / len(sc))
    R, Fl = np.array(pr['REAL']), np.array(pr['FALSE'])
    auc = np.mean([(r < f) + 0.5 * (r == f) for r in R for f in Fl])
    print(f'{opt:45s} median rank pct REAL {np.median(R):.3f} PL {np.median(pr["PLAUSIBLE"]):.3f} FALSE {np.median(Fl):.3f}   P(REAL ranked above FALSE) {auc:.2f}  REAL in top 10%: {sum(R<=0.1)}/{len(R)}, FALSE {sum(Fl<=0.1)}/{len(Fl)}')
