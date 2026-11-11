"""Target-machine smoke test (2 GPUs, TF2.15.1): real model, native AdamW, global batch 16.

Policy batch sizes (16,16,9,8,3,2) must give finite loss and weights.  Batch size 1 (one replica receives an EMPTY
batch) is run LAST as a diagnostic only: it reproduces/refutes the suspected cause of the V5 fold-1 NaN and is
reported, not asserted, because V6 never feeds such a batch (see batching.py).
"""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['FAT','Uniform_Transformer','Preaggregation_Transformer'],required=True);a=ap.parse_args()
from verify import verify
verify()
sys.path.insert(0,str(ROOT/a.model));sys.path.insert(1,str(ROOT))
import numpy as np,tensorflow as tf
if not hasattr(tf.optimizers,'AdamW') or not tf.config.list_physical_devices('GPU'):raise RuntimeError('Native AdamW and target GPUs required')
from fair_interfaces import seed_fold,aggregate_reference
from myloss import mask_loss_physical_gpp
from distributed_runtime import get_strategy,balanced_loss
strategy=get_strategy();seed_fold(42);rng=np.random.RandomState(42)
N=16
f=rng.normal(0,.2,(N,48,109)).astype(np.float32);f[:,:,5]=np.arange(N)[:,None]%2;f[:,:,96]=2.;f[:,:,3]=100;f[:,:,6]=np.arange(48)/2.
h=rng.normal(0,.2,(N,50,50,20)).astype(np.float32);w=rng.uniform(size=(N,48,50,50)).astype(np.float32)*1e-12
with strategy.scope():
    if a.model=='Preaggregation_Transformer':
        import Model_build_preagg_fairfix as mb,Model_history_preagg as mh
        m=mb.get_transformer_preagg(layer_cnn=2,layern=4,units=64,n_times=48,n_bands=6,n_flux_features=109,n_index=[69,68,70,71,72,73,75,76,77,78,80,64],n_head=4,drop=.1,is_mete=True,is_sensor=True,is_doy=True,is_hour=True,is_decoder=False)
        x=[aggregate_reference(h,w),f]
    else:
        import Model_build as mb,Model_history as mh
        m=mb.get_transformer_cnn_at(layer_cnn=2,layern=4,units=64,n_times=48,n_hls_features=20,n_bands=6,n_window=50,n_flux_features=109,n_index=[69,68,70,71,72,73,75,76,77,78,80,64],n_head=4,drop=.1,FFP=0 if a.model=='FAT' else 1,is_mete=True,is_sensor=True,is_doy=True,is_hour=True,is_decoder=False,is_degrade=False)
        x=[h,f,w]
    schedule=mh.WarmUpCosineDecay(start_lr=0.,target_lr=.0001,warmup_steps=5*100,total_steps=30*100,hold=0)
    m.compile(optimizer=tf.optimizers.AdamW(weight_decay=.1,learning_rate=schedule,beta_1=.9),loss=balanced_loss(mask_loss_physical_gpp(maskValue=-9999.,lambda1=1,beta=0)),metrics=[mh.Maskedaccuracy()])
y=np.zeros((N,48,4),np.float32);y[:,:,:3]=[1.,2.,3.]
def weights_finite():return all(np.isfinite(v).all() for v in m.get_weights())
report={'model':a.model,'tensorflow':tf.__version__,'replicas':strategy.num_replicas_in_sync,'steps':[],'full_training_completed':False}
for size,policy in [(16,True),(16,True),(9,True),(8,True),(3,True),(2,True),(1,False)]:
    xs=[v[:size] for v in x];entry={'batch':size,'policy_size':policy}
    try:
        r=np.asarray(m.train_on_batch(xs,y[:size]));entry['loss_finite']=bool(np.isfinite(r).all());entry['weights_finite']=bool(weights_finite());entry['values']=r.tolist()
    except Exception as e:
        entry['error']=repr(e)[:300];entry['loss_finite']=entry['weights_finite']=False
    report['steps'].append(entry);print(json.dumps(entry),flush=True)
    if policy and not (entry['loss_finite'] and entry['weights_finite']):
        raise RuntimeError('Policy batch %d produced nonfinite loss/weights: %s'%(size,entry))
    if not policy and not entry['weights_finite']:
        report['diagnostic']='batch=1 (empty replica batch) poisons the weights -> confirms the V5 fold-1 mechanism; V6 batching avoids it'
p=m.predict([v[:16] for v in x],batch_size=16,verbose=0)
if report['steps'][-1]['weights_finite']:assert np.isfinite(p).all(),'Nonfinite prediction'
report['policy_batches_finite']=True
(ROOT/'audit'/('target_smoke_'+a.model+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
