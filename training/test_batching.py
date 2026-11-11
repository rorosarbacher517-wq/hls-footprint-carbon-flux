"""Replica-safe batching test (runs on CPU with two logical devices; works on TF2.8 and TF2.15).

Checks: no record dropped, no step below one record per replica, last-batch rule, finite training with the
formerly fatal remainder (1 record), and prediction equal to one-shot prediction.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow as tf
cpu=tf.config.list_physical_devices('CPU')[0]
tf.config.set_logical_device_configuration(cpu,[tf.config.LogicalDeviceConfiguration(),tf.config.LogicalDeviceConfiguration()])
from batching import batch_sizes,make_batches,fit_with_safe_batches,predict_in_batches
from distributed_runtime import balanced_loss

for n,expected_tail in [(19905,[9,8]),(19846,[6]),(18000,[16]),(20616,[8]),(17,[9,8]),(33,[9,8]),(35,[3])]:
    s=batch_sizes(n,16,2)
    assert sum(s)==n and max(s)<=16 and min(s)>=2 and s[-len(expected_tail):]==expected_tail,(n,s[-3:])
    assert len(s)==-(-n//16),'steps per epoch must stay ceil(n/16)'
# every record exactly once per epoch, also after reshuffle
xs=[np.arange(50,dtype=np.float32)[:,None]]
seq=make_batches(xs,np.arange(50,dtype=np.float32)[:,None],batch_sizes(50,16,2),True,seed=1)
for _ in range(2):
    seen=np.concatenate([seq[i][1][:,0] for i in range(len(seq))]);assert sorted(seen)==list(range(50));seq.on_epoch_end()

strategy=tf.distribute.MirroredStrategy(['CPU:0','CPU:1'])
with strategy.scope():
    a=tf.keras.Input((4,));b=tf.keras.Input((3,))
    h=tf.keras.layers.Dense(8,activation='relu')(tf.keras.layers.Concatenate()([a,b]))
    bn=tf.keras.layers.experimental.SyncBatchNormalization() if tf.__version__.startswith('2.8') else tf.keras.layers.BatchNormalization(synchronized=True)
    h=bn(h)  # synchronized BN, as in production (non-sync moving stats differ per replica)
    model=tf.keras.Model([a,b],tf.keras.layers.Dense(2)(h))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=balanced_loss(lambda y,p:tf.reduce_mean(tf.square(y-p),axis=-1)))
rng=np.random.RandomState(0)
n=85                                 # int(85*.96)=81 -> 81 % 16 == 1 : formerly one EMPTY replica batch
assert int(n*(1-0.04))%16==1
X=[rng.randn(n,4).astype('f'),rng.randn(n,3).astype('f')];Y=rng.randn(n,2).astype('f')
np.random.seed(3)
hist=fit_with_safe_batches(model,X,Y,2,16,0.04,[])
assert np.isfinite(hist.history['loss']).all() and np.isfinite(hist.history['val_loss']).all()
assert all(np.isfinite(w).all() for w in model.get_weights())
p1=predict_in_batches(model,X,16,verbose=0);p2=model.predict(X,batch_size=n,verbose=0)
assert p1.shape==(n,2) and np.allclose(p1,p2,atol=1e-5)
print('PASS batching: no dropped records, no empty replica batch, finite training, prediction identical')
