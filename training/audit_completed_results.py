"""After training: require all 20 held-out predictions (4 models x 5 folds) on frozen observations.

No silent NaN filtering and no mixture of releases/environments/epochs.
Five-fold SD uses ddof=1; pooled metrics are separately labeled.
Writes one combined comparison table (comparison_table.csv / .md).
Smoke roots (--smoke --epochs N) are audited for plumbing only and written to a separate folder.
"""
from pathlib import Path
import argparse,json,csv
import numpy as np
from numpy_pickle_compat import install_numpy_pickle_compat
install_numpy_pickle_compat()
ROOT=Path(__file__).resolve().parent
MODELS=['FAT','Uniform_Transformer','Preaggregation_Transformer','RF']
FLUX=['NEE','GPP','RECO']

def metric(y,p):
    """Bias is observed - predicted everywhere."""
    y=np.asarray(y,np.float64);p=np.asarray(p,np.float64)
    den=np.sum((y-y.mean())**2)
    return {'R2':float(1-np.sum((p-y)**2)/den) if den>0 else float('nan'),'RMSE':float(np.sqrt(np.mean((p-y)**2))),'Bias':float(np.mean(y-p)),'n':int(len(y))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-root',type=Path,default=Path('./data/external/results_v6_tf215'))
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--epochs',type=int)
    ap.add_argument('--models',help='comma list; default all four. A subset writes to a *_partial folder and is NOT the final paper table')
    a=ap.parse_args()
    models=a.models.split(',') if a.models else list(MODELS)
    if any(m not in MODELS for m in models):raise RuntimeError('unknown model in --models')
    partial=set(models)!=set(MODELS)
    cfg=json.loads((ROOT/'config.json').read_text())
    epochs=int(cfg['dl']['epochs'])
    if a.smoke:
        if a.epochs is None or 'smoke' not in str(a.results_root).lower():raise RuntimeError('--smoke needs --epochs and a smoke results root')
        epochs=a.epochs
    elif 'smoke' in str(a.results_root).lower():raise RuntimeError('A smoke root cannot be audited as production')
    from verify import verify,sha
    verify()
    expected={'release':cfg['version'],'code_lock_sha256':sha(ROOT/'audit/code_lock.json'),'input_manifest_sha256':sha(ROOT/'reference/input_manifest.json'),'epochs':epochs,'smoke':bool(a.smoke)}
    from result_contract import validate
    env_path=a.results_root/'environment.json'
    if not env_path.exists():raise RuntimeError('Missing environment.json: '+str(env_path))
    expected['environment']=json.loads(env_path.read_text())
    rows=[];merged={};reference={};alignment_reference={}
    for name in models:
        base_root=a.results_root/name;identity=base_root/'execution_identity.json'
        if not identity.exists() or json.loads(identity.read_text())!=expected:raise RuntimeError('Wrong/missing experiment release identity: '+name)
        ys=[];ps=[]
        for k in range(5):
            record=validate(ROOT,name,k,expected,a.results_root,None if name=='RF' else epochs)
            if k in alignment_reference and record!=alignment_reference[k]:raise RuntimeError('Cross-model preprocessing/split mismatch: '+name)
            alignment_reference[k]=record
            found=list(base_root.glob('**/fold%d/pred/true.npy'%k))
            if len(found)!=1:raise RuntimeError('Need exactly one result: %s fold%d, found %d'%(name,k,len(found)))
            pred_dir=found[0].parent;y=np.load(pred_dir/'true.npy');p=np.load(pred_dir/'pred.npy');site=np.load(pred_dir/'site.npy',allow_pickle=True)
            effective=np.load(ROOT/'reference'/('fold%d_effective_indices.npz'%k))['test']
            align=base_root/'alignment'/('fold%d_indices.npz'%k)
            if not align.exists() or not np.array_equal(np.load(align)['test'],effective):raise RuntimeError('Missing/mismatched run identity')
            if y.shape!=p.shape or y.shape!=(len(effective),48,3):raise RuntimeError('Shape mismatch: %s fold%d: %s %s'%(name,k,y.shape,p.shape))
            valid=np.isfinite(y)&(y!=-9999.)
            if np.any(valid&~np.isfinite(p)):raise RuntimeError('Nonfinite labeled predictions: '+name)
            canonical=np.where(valid,y,0.)
            if k not in reference:reference[k]=(valid,canonical,site)
            else:
                rv,ry,rs=reference[k]
                if not np.array_equal(valid,rv) or not np.array_equal(canonical,ry) or not np.array_equal(site,rs):raise RuntimeError('Observation/site ordering mismatch')
            for i,var in enumerate(FLUX):rows.append(dict(model=name,fold=k,flux=var,**metric(y[...,i][valid[...,i]],p[...,i][valid[...,i]])))
            ys.append(y);ps.append(p)
        merged[name]=(np.concatenate(ys),np.concatenate(ps))
    target=ROOT/'audit'/(('completed_results_smoke' if a.smoke else 'completed_results')+('_partial' if partial else ''));target.mkdir(parents=True,exist_ok=True)
    def save(fname,values):
        with (target/fname).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    save('fold_metrics.csv',rows);summary=[];pooled=[]
    for name in models:
        y,p=merged[name];valid=np.isfinite(y)&(y!=-9999.)
        for i,var in enumerate(FLUX):
            five=[x for x in rows if x['model']==name and x['flux']==var];rec={'model':name,'flux':var,'n':sum(x['n'] for x in five)}
            for key in ['R2','RMSE','Bias']:
                z=np.array([x[key] for x in five],float);rec[key+'_mean']=float(z.mean());rec[key+'_sd']=float(z.std(ddof=1))
            summary.append(rec);pooled.append(dict(model=name,flux=var,**metric(y[...,i][valid[...,i]],p[...,i][valid[...,i]])))
    save('fivefold_mean_sd.csv',summary);save('pooled_metrics.csv',pooled)
    # One combined table: rows = model, columns = flux x (RMSE, R2, Bias) as mean +/- SD over 5 folds.
    table=[]
    for name in models:
        row={'model':name}
        for var in FLUX:
            r=next(x for x in summary if x['model']==name and x['flux']==var)
            for key in ['RMSE','R2','Bias']:row['%s_%s'%(var,key)]='%.3f +/- %.3f'%(r[key+'_mean'],r[key+'_sd'])
        table.append(row)
    save('comparison_table.csv',table)
    cols=list(table[0]);md=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(r[c] for c in cols)+' |' for r in table]
    (target/'comparison_table.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
    print('PASS: %d folds share held-out observation identity%s%s; metrics saved to %s'%(5*len(models),' (SMOKE: plumbing only, not scientific results)' if a.smoke else '',' (PARTIAL model set: not the final four-model table)' if partial else '',target))
if __name__=='__main__':main()
