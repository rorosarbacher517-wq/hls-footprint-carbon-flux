"""Auditable interfaces; corrected shared data/weight semantics; FAT network and fit retained."""
import os,json,hashlib,random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
COVARIATES=[3,5,6,69,68,70,71,72,73,75,76,77,78,80,64]

def digest_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()

def seed_fold(seed):
    import tensorflow as tf
    random.seed(seed);np.random.seed(seed);tf.random.set_seed(seed)

def host_inputs(function,*args,**kwargs):
    # Placement only: preserve the original conversion/dtypes and sample order.
    import tensorflow as tf
    hls=args[0]
    for start in range(0,len(hls),128):
        h=np.asarray(hls[start:start+128,...,:6])
        valid=np.all(np.isfinite(h)&(h!=-9999.),axis=-1)
        if not np.all(np.any(valid,axis=(1,2))):
            raise RuntimeError('No valid six-band HLS pixel; refusing sample dropping')
    with tf.device('/CPU:0'):
        return function(*args,**kwargs)

def aggregate_reference(hls,ffp,chunk_size=16):
    """Use the actual FAT attention function, including stable normalization and valid-pixel fallback.

    HLS must ALREADY be normalized by the original FAT normalizer. FAT and Uniform normalize pixels first, preserving one common interface.
    The corrected weights sum to one on their valid support.
    PREAGG and RF call this exact function on identical six-band arrays.
    """
    import tensorflow as tf
    from model_heads_ffp import build_true_dynamic_ffp_attention
    n=int(hls.shape[0]);t=int(ffp.shape[1])
    out=np.empty((n,t,6),np.float32)
    with tf.device('/CPU:0'):
        for start in range(0,n,chunk_size):
            stop=min(start+chunk_size,n)
            h=tf.convert_to_tensor(hls[start:stop,...,:6],dtype=tf.float32)
            f=tf.convert_to_tensor(ffp[start:stop],dtype=tf.float32)
            valid=tf.reduce_all(tf.math.is_finite(h)&tf.not_equal(h,-9999.),axis=-1,keepdims=True)
            weights=build_true_dynamic_ffp_attention(f,tf.cast(valid,tf.float32),is_degrade=False,fallback_mode='center1')
            clean=tf.where(valid,h,tf.zeros_like(h))
            out[start:stop]=tf.einsum('bthw,bhwc->btc',weights,clean).numpy()
    return out

def audit_fold(fold,outer_train,outer_test,keep_train,keep_test,flux_train,flux_test):
    for role,idx in [('train',outer_train),('test',outer_test)]:
        ref=np.load(ROOT/'reference/split_kfold'/('index_%s_fold%d.npy'%(role,fold)))
        if not np.array_equal(idx,ref):raise RuntimeError('Frozen outer split mismatch: '+role)
    tr=np.flatnonzero(outer_train)[keep_train];te=np.flatnonzero(outer_test)[keep_test]
    cut=int(len(tr)*.96)
    pack={'train':tr[:cut],'validation':tr[cut:],'test':te,'outer_train_retained':tr}
    frozen=np.load(ROOT/'reference'/('fold%d_effective_indices.npz'%fold))
    for role,rows in pack.items():
        if not np.array_equal(rows,frozen[role]):raise RuntimeError('Effective record/order mismatch: '+role)
    record={k:{'n':len(v),'sha256':digest_array(v)} for k,v in pack.items()}
    for role,flux in [('train',flux_train),('test',flux_test)]:
        f=flux.numpy() if hasattr(flux,'numpy') else np.asarray(flux)
        sensor=f[...,5]
        if not np.all(np.isfinite(sensor)&((sensor==0)|(sensor==1))):
            raise RuntimeError('Sensor must be raw integer category 0/1 at every input time')
        y=f[...,7:10];valid=np.isfinite(y)&(y!=-9999.)
        if not np.all(np.isfinite(y)):raise RuntimeError('Nonfinite labels differ from legacy loss mask')
        if not (np.array_equal(valid[...,0],valid[...,1]) and np.array_equal(valid[...,0],valid[...,2])):
            raise RuntimeError('Target masks differ; RF common mask would change the comparison')
        x=f[...,COVARIATES]
        if np.any(valid[...,0]&~np.all(np.isfinite(x)&(x!=-9999.),axis=-1)):
            raise RuntimeError('RF would reject labeled covariate rows; fairness check failed')
        record[role+'_flux_sha256']=digest_array(f)
        record[role+'_valid_per_target']=valid.sum(axis=(0,1)).tolist()
    out=Path(os.environ['TRAIN_DATA_ROOT'])/'alignment';out.mkdir(parents=True,exist_ok=True)
    np.savez(out/('fold%d_indices.npz'%fold),**pack)
    path=out/('fold%d.json'%fold)
    path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    # Refuse a disagreement against any previously run model; never auto-repair it.
    for other in Path(os.environ['TRAIN_DATA_ROOT']).parent.glob('*/alignment/fold%d.json'%fold):
        if other!=path and json.loads(other.read_text(encoding='utf-8'))!=record:
            raise RuntimeError('Actual model data disagree: '+str(other))
    return record


def validate_predictions(y_true,y_pred):
    y=y_true.numpy() if hasattr(y_true,'numpy') else np.asarray(y_true)
    p=y_pred.numpy() if hasattr(y_pred,'numpy') else np.asarray(y_pred)
    if y.shape!=p.shape:raise RuntimeError('Prediction/target shape mismatch')
    valid=np.isfinite(y)&(y!=-9999.)
    if np.any(valid&~np.isfinite(p)):
        raise RuntimeError('Nonfinite predictions on labeled observations: refusing silent evaluation sample deletion')
    return True


# ---- per-fold host-memory release (V5 smoke: fold 2 OOM-killed, previous fold's arrays were still referenced) ----
_PROTECTED=set()

def _is_big(v):
    if isinstance(v,np.ndarray):return v.nbytes>50_000_000
    shape=getattr(v,'shape',None);dtype=getattr(v,'dtype',None)
    if shape is not None and dtype is not None and hasattr(v,'numpy'):
        try:return int(np.prod(shape))*np.dtype(dtype.as_numpy_dtype).itemsize>50_000_000
        except Exception:return False
    return False

def protect_inputs(namespace):
    """Call once after loading the three input arrays: they must survive every fold."""
    _PROTECTED.update(id(v) for v in namespace.values() if _is_big(v))

def release_fold_memory(namespace):
    """Call at the top of each fold: drop the previous fold's large arrays/model, keep protected inputs."""
    import gc
    for name in ('model','model_history'):namespace.pop(name,None)
    for name,value in list(namespace.items()):
        if not name.startswith('__') and id(value) not in _PROTECTED and _is_big(value):del namespace[name]
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:pass
    gc.collect()
