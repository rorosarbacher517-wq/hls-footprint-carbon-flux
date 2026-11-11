"""Two logical CPUs validate replica reductions; not a two-GPU performance test."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow as tf
cpu=tf.config.list_physical_devices('CPU')[0]
tf.config.set_logical_device_configuration(cpu,[tf.config.LogicalDeviceConfiguration(),tf.config.LogicalDeviceConfiguration()])
from distributed_runtime import balanced_loss
strategy=tf.distribute.MirroredStrategy(devices=['/CPU:0','/CPU:1'])
with strategy.scope():
    weight=tf.Variable(2.)
    wrapped=tf.keras.losses.LossFunctionWrapper if hasattr(tf.keras.losses,'LossFunctionWrapper') else None
    class Loss(tf.keras.losses.Loss):
        def call(self,y,p):return balanced_loss(lambda a,b:tf.reduce_mean(tf.square(a-b),axis=1))(y,p)
    loss=Loss(reduction=tf.keras.losses.Reduction.NONE)
    if tf.__version__.startswith('2.8'):
        bn=tf.keras.layers.experimental.SyncBatchNormalization(momentum=.9)
    else:bn=tf.keras.layers.BatchNormalization(synchronized=True,momentum=.9)
    bn.build((None,1))
@tf.function
def step(x):
    def replica(z):
        with tf.GradientTape() as tape:
            values=loss(tf.zeros_like(z),z*weight)
            v=tf.math.divide_no_nan(tf.reduce_sum(values),tf.cast(tf.size(values),tf.float32))/2.
        grad=tape.gradient(v,weight)
        normalized=bn(z,training=True)
        return v,grad,normalized
    return strategy.run(replica,args=(x,))
for n in [16,6,5,1]:
    x=np.arange(1,n+1,dtype=np.float32).reshape(-1,1);cut=(n+1)//2
    dist=strategy.experimental_distribute_values_from_function(lambda ctx:tf.constant(x[:cut] if ctx.replica_id_in_sync_group==0 else x[cut:]))
    v,g,z=step(dist)
    value=strategy.reduce(tf.distribute.ReduceOp.SUM,v,axis=None).numpy()
    gradient=strategy.reduce(tf.distribute.ReduceOp.SUM,g,axis=None).numpy()
    np.testing.assert_allclose(value,np.mean((x*2)**2),rtol=1e-6)
    np.testing.assert_allclose(gradient,np.mean(4*x*x),rtol=1e-6)
    got=np.concatenate([t.numpy() for t in strategy.experimental_local_results(z)])
    expected=(x-x.mean())/np.sqrt(x.var()+bn.epsilon)
    np.testing.assert_allclose(got,expected,atol=2e-6)
    print('PASS global loss, gradient, synchronized BN; global batch',n,flush=True)
print('PASS two-logical-CPU math tests. Actual TF2.15.1 two-A100 training still required.')
