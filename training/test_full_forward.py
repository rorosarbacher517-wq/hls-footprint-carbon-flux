"""Full-architecture forward test only; no fit/optimizer updates or batch-policy change."""
import sys,json,importlib.util,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'FAT'))
import numpy as np,tensorflow as tf
from fair_interfaces import seed_fold
import Model_build as mb
rng=np.random.RandomState(42)
h=np.broadcast_to(rng.normal(0,.1,(2,1,1,20)).astype(np.float32),(2,50,50,20)).copy()
f=np.zeros((2,48,109),np.float32);f[1,:,5]=1;f[:,:,3]=100;f[:,:,6]=np.arange(48)/2.;f[:,:,96]=np.linspace(.5,2.5,48,dtype=np.float32)[None,:]
w=rng.uniform(size=(2,48,50,50)).astype(np.float32)*1e-14
args=dict(layer_cnn=2,layern=4,decoder_layer_n=2,units=64,n_times=48,n_hls_features=20,n_bands=6,n_window=50,n_head=4,drop=.1,n_out=3,n_flux_features=109,n_index=[69,68,70,71,72,73,75,76,77,78,80,64],is_batch=True,mask_value=-9999.,is_mete=True,is_sensor=True,is_xxy=False,is_doy=True,is_hour=True,final_relu=False,is_decoder=False,is_degrade=False,radius=.03)
tf.keras.backend.clear_session();seed_fold(42);m=mb.get_transformer_cnn_at(FFP=0,**args)
a=m([h,f,w],training=False).numpy();assert a.shape==(2,48,3) and np.isfinite(a).all();del m;gc.collect()
us=importlib.util.spec_from_file_location('uniform_heads',ROOT/'Uniform_Transformer/model_heads_ffp.py');um=importlib.util.module_from_spec(us);us.loader.exec_module(um);mb.build_weight_head=um.build_weight_head
tf.keras.backend.clear_session();seed_fold(42);m=mb.get_transformer_cnn_at(FFP=1,**args)
b=m([h,f,w],training=False).numpy();assert b.shape==(2,48,3) and np.isfinite(b).all();del m;gc.collect()
sys.path.insert(0,str(ROOT/'Preaggregation_Transformer'));import Model_build_preagg_fairfix as pm
from fair_interfaces import aggregate_reference
hp=aggregate_reference(h,w)
tf.keras.backend.clear_session();seed_fold(42);m=pm.get_transformer_preagg(layer_cnn=2,layern=4,decoder_layer_n=2,units=64,n_times=48,n_bands=6,n_head=4,drop=.1,n_out=3,n_flux_features=109,is_mete=True,is_sensor=True,is_doy=True,is_hour=True,is_decoder=False)
p=m([hp,f],training=False).numpy();assert p.shape==(2,48,3) and np.isfinite(p).all()
report={'all_three_full_architectures_finite':True,'shape':[2,48,3],'FAT_uniform_timevarying_uniform_forward_finite':True,'tiny_weight_scale':1e-14,'inference_batch':2,'configured_training_batch_unchanged':16,'optimizer_updates':0,'tensorflow':tf.__version__}
(ROOT/'audit/full_forward_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
