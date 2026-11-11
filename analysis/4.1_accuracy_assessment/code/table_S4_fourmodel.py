"""Section 4.1 supplementary table (Table S4 style): combine four models' five-fold held-out predictions into one Table-S4-style comparison.

Input : ../../data/four_model_predictions/<Model>/fold{0-4}/{true,pred,site}.npy  (origin of every file: data/four_model_predictions/SOURCES.json)
Output: ../outputs/comparison_table_S4_style.csv / .md, fold_metrics.csv, pooled_metrics.csv

Checks (hard failures): identical held-out observations and site/time order in all four models, no non-finite
prediction at a labeled observation.  Metrics per fold on valid observations; table = mean +/- SD over the 5 folds
(ddof=1).  R2 = 1 - SSE/SST; RMSE; Bias = predicted - observed (same convention as the original Table S4:
the original FAT reproduces its published signs with this definition).  n = pooled number of valid observations.
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / 'data' / 'four_model_predictions'
OUT = HERE.parent / 'outputs'
OUT.mkdir(exist_ok=True)
MODELS = [  # (result folder, model name in table = same names as the figure; aggregation is part of the name)
    ('FAT', 'FAT', ''),
    ('Uniform_Transformer', 'Uniform-weighted Transformer', ''),
    ('Preaggregation_Transformer', 'Pre-agg. Transformer', ''),
    ('RF', 'Pre-agg. RF', ''),
]
FLUX = ['NEE', 'GPP', 'RECO']


def metrics(y, p):
    den = np.sum((y - y.mean()) ** 2)
    return {'R2': 1 - np.sum((p - y) ** 2) / den, 'RMSE': np.sqrt(np.mean((p - y) ** 2)), 'Bias': np.mean(p - y)}


def load(model, fold):
    base = DATA / model / ('fold%d' % fold)
    return {k: np.load(base / (k + '.npy'), allow_pickle=True) for k in ('true', 'pred', 'site')}


def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    rows, pooled, ref = [], [], {}
    pooled_y = {m: [] for m, _, _ in MODELS}
    pooled_p = {m: [] for m, _, _ in MODELS}
    for model, _, _ in MODELS:
        for k in range(5):
            d = load(model, k)
            y, p = d['true'], d['pred']
            valid = np.isfinite(y) & (y != -9999.)
            if y.shape != p.shape or y.shape[1:] != (48, 3):
                raise RuntimeError('shape mismatch %s fold%d %s %s' % (model, k, y.shape, p.shape))
            if np.any(valid & ~np.isfinite(p)):
                raise RuntimeError('non-finite prediction at a labeled observation: %s fold%d' % (model, k))
            canon = np.where(valid, y, 0.)
            if k not in ref:
                ref[k] = (valid, canon, d['site'])
            elif not (np.array_equal(valid, ref[k][0]) and np.array_equal(canon, ref[k][1]) and np.array_equal(d['site'], ref[k][2])):
                raise RuntimeError('held-out observation/site order differs: %s fold%d' % (model, k))
            for i, fx in enumerate(FLUX):
                m = metrics(y[..., i][valid[..., i]].astype(np.float64), p[..., i][valid[..., i]].astype(np.float64))
                rows.append(dict(model=model, fold=k, flux=fx, n=int(valid[..., i].sum()), **m))
            pooled_y[model].append(y)
            pooled_p[model].append(p)
    for model, _, _ in MODELS:
        y, p = np.concatenate(pooled_y[model]), np.concatenate(pooled_p[model])
        valid = np.isfinite(y) & (y != -9999.)
        for i, fx in enumerate(FLUX):
            pooled.append(dict(model=model, flux=fx, n=int(valid[..., i].sum()),
                               **metrics(y[..., i][valid[..., i]].astype(np.float64), p[..., i][valid[..., i]].astype(np.float64))))

    def write(name, data):
        with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    write('fold_metrics.csv', rows)
    write('pooled_metrics.csv', pooled)
    table = []
    for fx in FLUX:
        for model, label, agg in MODELS:
            sel = [r for r in rows if r['model'] == model and r['flux'] == fx]
            rec = {'Flux': fx, 'Model': label}
            for key, nd in (('R2', 2), ('RMSE', 2), ('Bias', 2)):
                v = np.array([r[key] for r in sel])
                rec['R²' if key == 'R2' else key] = '%.*f ± %.*f' % (nd, v.mean(), nd, v.std(ddof=1))
            rec['n'] = sum(r['n'] for r in sel)
            table.append(rec)
    write('comparison_table_S4_style.csv', table)
    cols = list(table[0])
    md = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    md += ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for r in table]
    (OUT / 'comparison_table_S4_style.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
