"""Result receipts: issued only by the launcher after successful subprocess exit."""
import json
from pathlib import Path
import numpy as np
from verify import sha

ROLES=('train','validation','test','outer_train_retained')

def alignment(root,name,k,results_root=None):
    results_root=Path(results_root) if results_root is not None else root/'results'
    base=results_root/name/'alignment'
    with np.load(base/('fold%d_indices.npz'%k)) as actual, np.load(root/'reference'/('fold%d_effective_indices.npz'%k)) as expected:
        for role in ROLES:
            if not np.array_equal(actual[role],expected[role]):
                raise RuntimeError('Frozen index mismatch: %s fold%d %s'%(name,k,role))
    record=json.loads((base/('fold%d.json'%k)).read_text(encoding='utf-8'))
    for key in ('train_flux_sha256','test_flux_sha256'):
        if not isinstance(record.get(key),str) or len(record[key])!=64:
            raise RuntimeError('Missing preprocessing fingerprint: '+key)
    return record

def artifacts(root,name,k,results_root=None,epochs=None):
    results_root=Path(results_root) if results_root is not None else root/'results'
    base=results_root/name
    found=list(base.glob('**/fold%d/pred/true.npy'%k))
    if len(found)!=1:raise RuntimeError('Missing/ambiguous fold artifacts')
    fold=found[0].parent.parent
    files=[fold/'pred'/v for v in ('true.npy','pred.npy','site.npy')]
    files += [base/'alignment'/('fold%d%s'%(k,suffix)) for suffix in ('_indices.npz','.json')]
    if name=='RF':
        for flux in ('NEE','GPP','RECO'):
            files += [fold/'model'/flux/v for v in ('rf_model.joblib','spec.json')]
    else:
        history=fold/'model/history.json'
        h=json.loads(history.read_text(encoding='utf-8'))
        epochs=epochs if epochs is not None else json.loads((root/'config.json').read_text())['dl']['epochs']
        for key in ('loss','val_loss'):
            values=np.asarray(h.get(key,[]),dtype=float)
            if values.shape!=(epochs,) or not np.isfinite(values).all():
                raise RuntimeError('Incomplete/nonfinite training history: '+str(history))
        files += [fold/'model'/v for v in ('history.json','weights.h5','spec.json')]
    for path in files:
        if not path.is_file() or path.stat().st_size==0:raise RuntimeError('Missing/empty artifact: '+str(path))
    return fold,{p.relative_to(base).as_posix():sha(p) for p in files}

def certify(root,name,folds,results_root=None,epochs=None):
    results_root=Path(results_root) if results_root is not None else root/'results'
    base=results_root/name
    identity=json.loads((base/'execution_identity.json').read_text())
    pending=[]
    for k in folds:
        alignment(root,name,k,results_root)
        fold,hashes=artifacts(root,name,k,results_root,epochs)
        pending.append((fold,dict(status='complete',model=name,fold=k,identity=identity,artifacts=hashes)))
    for fold,receipt in pending:
        temp=fold/'completion.json.tmp'
        temp.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        temp.replace(fold/'completion.json')

def validate(root,name,k,identity,results_root=None,epochs=None):
    record=alignment(root,name,k,results_root)
    fold,hashes=artifacts(root,name,k,results_root,epochs)
    receipt=json.loads((fold/'completion.json').read_text(encoding='utf-8'))
    expected=dict(status='complete',model=name,fold=k,identity=identity,artifacts=hashes)
    if receipt!=expected:raise RuntimeError('Missing/stale completion receipt: '+str(fold))
    return record
