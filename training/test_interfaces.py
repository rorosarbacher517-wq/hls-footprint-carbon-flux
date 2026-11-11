"""Strict interface/geometry checks; no production training."""
import os,sys,json,importlib.util,gc
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'FAT'))
import numpy as np,tensorflow as tf
from fair_interfaces import aggregate_reference,seed_fold
from height_uniform import make_tensorflow_layer,height_uniform_weights_numpy
import model_heads_ffp as corrected_heads

rng=np.random.RandomState(917)
h=rng.normal(size=(4,50,50,6)).astype(np.float32)
f=rng.uniform(size=(4,48,50,50)).astype(np.float32)
f[0]*=1e-14
f[1]=0.
f[2,:,2:5,3:7]=np.nan
h[2,2:5,3:7]=-9999.;h[3,25,25]=-9999.;f[3]=0
valid=np.all(np.isfinite(h)&(h!=-9999.),axis=-1,keepdims=True)
w=corrected_heads.build_true_dynamic_ffp_attention(tf.constant(f),tf.constant(valid,dtype=tf.float32),is_degrade=False).numpy()
expected=tf.einsum('bthw,bhwc->btc',w,np.where(valid,h,0)).numpy()
oracle=np.zeros(f.shape,np.float64)
for i in range(len(h)):
    v=valid[i,...,0]
    for j in range(f.shape[1]):
        w0=np.where(np.isfinite(f[i,j])&(f[i,j]>0)&v,f[i,j],0).astype(np.float64)
        if w0.sum()==0:
            if v[25,25]:w0[25,25]=1.
            else:w0=v.astype(np.float64)
        oracle[i,j]=w0/w0.sum()
np.testing.assert_allclose(w,oracle,rtol=2e-6,atol=1e-7)
np.testing.assert_allclose(w.sum(axis=(2,3)),1.,rtol=0,atol=1e-6)
assert np.all(w[~np.broadcast_to(valid[...,0][:,None],w.shape)]==0)
scaled=corrected_heads.build_true_dynamic_ffp_attention(tf.constant(f*1e-8),tf.constant(valid,dtype=tf.float32)).numpy()
np.testing.assert_allclose(scaled,w,rtol=2e-6,atol=1e-7)
a=aggregate_reference(h,f,chunk_size=4);b=aggregate_reference(h,f,chunk_size=1)
np.testing.assert_array_equal(a,expected);np.testing.assert_allclose(b,a,rtol=2e-6,atol=2e-6)

# Half-hourly Uniform: each timestep can have its own effective height.
flux=np.zeros((4,48,109),np.float32)
flux[:,:,96]=1.0
flux[1,:,96]=np.linspace(0.45,2.50,48,dtype=np.float32)  # deliberately changes snapped support
flux[2,:,96]=np.nan
flux[2,:,102]=np.linspace(0.60,1.60,48,dtype=np.float32)  # same-timestep smooth fallback
flux[3,:,96]=20.0
valid_all=np.ones((4,50,50),dtype=bool)
usp=importlib.util.spec_from_file_location('uniform_heads',ROOT/'Uniform_Transformer/model_heads_ffp.py')
um=importlib.util.module_from_spec(usp);usp.loader.exec_module(um)
un=um.HeightUniformAttention()([tf.constant(valid_all[...,None],dtype=tf.float32),tf.constant(flux)]).numpy()
nw=height_uniform_weights_numpy(valid_all,flux)
assert un.shape==(4,48,50,50)
np.testing.assert_allclose(un,nw,rtol=1e-6,atol=1e-7)
assert np.allclose(un.sum(axis=(2,3)),1.,rtol=0,atol=1e-6)
assert (un[0,0]>0).sum()==36
assert (un[3,0]>0).sum()==2500
assert (un[1,0]>0).sum()!= (un[1,-1]>0).sum(), 'Uniform must remain timestep-specific'
assert not np.array_equal(un[1,0],un[1,-1]), 'No record-level weight tiling is allowed'
assert np.array_equal(un[2,0],nw[2,0]) and np.array_equal(un[2,-1],nw[2,-1])
assert np.array_equal(un[2,0],un[2,0])

# Build real architecture and confirm FAT/Uniform parameter identity.
import Model_build as mb
args=dict(layer_cnn=2,layern=4,decoder_layer_n=2,units=64,n_times=48,n_hls_features=20,n_bands=6,n_window=50,n_head=4,drop=.1,n_out=3,n_flux_features=109,n_index=[69,68,70,71,72,73,75,76,77,78,80,64],is_batch=True,mask_value=-9999.,is_mete=True,is_sensor=True,is_xxy=False,is_doy=True,is_hour=True,final_relu=False,is_decoder=False,is_degrade=False,radius=.03)
tf.keras.backend.clear_session();seed_fold(42);model=mb.get_transformer_cnn_at(FFP=0,**args)
fat_weights=[v.numpy().copy() for v in model.weights];fat_count=model.count_params()
mb.build_weight_head=um.build_weight_head
tf.keras.backend.clear_session();seed_fold(42);u=mb.get_transformer_cnn_at(FFP=1,**args)
for old,new in zip(fat_weights,u.weights):np.testing.assert_array_equal(old,new.numpy())
assert fat_count==u.count_params()==223427

sys.path.insert(0,str(ROOT/'Preaggregation_Transformer'))
import Model_build_preagg_fairfix as pb
tf.keras.backend.clear_session();seed_fold(42);p=pb.get_transformer_preagg(layer_cnn=2,layern=4,decoder_layer_n=2,units=64,n_times=48,n_bands=6,n_head=4,drop=.1,n_out=3,n_flux_features=109,is_mete=True,is_sensor=True,is_doy=True,is_hour=True,is_decoder=False)
report={
 'dynamic_attention_oracle_max_abs_diff':float(abs(w-oracle).max()),
 'dynamic_unit_mass_max_abs_error':float(abs(w.sum(axis=(2,3))-1).max()),
 'reference_attention_max_abs_diff':float(abs(a-expected).max()),
 'chunking_max_abs_diff':float(abs(a-b).max()),
 'uniform_numpy_tensorflow_max_abs_diff':float(abs(un-nw).max()),
 'uniform_weights_shape':list(un.shape),
 'uniform_is_timestep_specific':True,
 'FAT_uniform_initial_weights_exact':True,
 'FAT_uniform_parameter_count':fat_count,
 'PREAGG_parameter_count':p.count_params(),
 'sensor_semantics':'per-timestep categorical 0/1; no standardization and no within-record constancy assumption',
 'synthetic_cases':['tiny positive FFP','zero FFP','NaN FFP','invalid HLS','invalid centre fallback','48-step time-varying uniform height','same-timestep zm_smooth fallback','grid snapping and clipping'],
 'training_performed':False,
 'tensorflow_test_version':tf.__version__
}
(ROOT/'audit/interface_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
