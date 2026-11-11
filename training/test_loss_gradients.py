import os,sys,json,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'FAT'))
import numpy as np,tensorflow as tf
from myloss import mask_loss_physical_gpp
sp=importlib.util.spec_from_file_location('original_loss',ROOT/'reference/original_code/FAT/myloss.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
new=mask_loss_physical_gpp(lambda1=1,beta=0);original=old.mask_loss_physical_gpp(lambda1=1,beta=0)
rng=np.random.RandomState(18);p=tf.Variable(rng.normal(size=(16,48,3)).astype(np.float32));y=tf.constant(rng.normal(size=(16,48,4)).astype(np.float32))
with tf.GradientTape() as t:a=tf.reduce_sum(new(y,p))
ga=t.gradient(a,p).numpy()
with tf.GradientTape() as t:b=tf.reduce_sum(original(y,p))
gb=t.gradient(b,p).numpy()
np.testing.assert_array_equal(a.numpy(),b.numpy());np.testing.assert_array_equal(ga,gb)
report={'nonzero_loss_values_exact':True,'nonzero_loss_gradients_exact':True,'zero_cases':[]}
for target,pred in [([1.,2.,3.,0.],[1.,2.,3.]),([-9999.,-9999.,-9999.,0.],[1.,2.,3.]),([2.,3.,4.,0.],[1.,2.,3.])]:
    v=tf.Variable([[pred]],dtype=tf.float32);yy=tf.constant([[target]],dtype=tf.float32)
    with tf.GradientTape() as t:l=tf.reduce_sum(new(yy,v))
    g=t.gradient(l,v).numpy();assert np.isfinite(g).all() and np.isfinite(l.numpy())
    report['zero_cases'].append({'target':target,'prediction':pred,'loss':float(l.numpy()),'gradient':g.tolist()})
(ROOT/'audit/loss_gradient_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
