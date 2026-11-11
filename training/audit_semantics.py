import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'FAT'))
import numpy as np,tensorflow as tf
from model_heads_ffp import build_true_dynamic_ffp_attention
c=json.loads((ROOT/'config.json').read_text());inp=Path(c['input_root'])
h=np.load(inp/'HLS_matched_v11_remove_selected_sites.npy',mmap_mode='r');f=np.load(inp/'FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy',mmap_mode='r')
idx=np.concatenate([np.load(ROOT/'reference'/('fold%d_effective_indices.npz'%k))['test'] for k in range(5)])
maximum=0.;times=0;invalid_mass=0
for start in range(0,len(idx),64):
    rows=idx[start:start+64];hs=np.transpose(np.array(h[rows,:6]),(0,2,3,1));ff=np.array(f[rows]);valid=np.all(np.isfinite(hs)&(hs!=-9999.),axis=-1,keepdims=True)
    assert np.all(np.any(valid,axis=(1,2,3)))
    w=build_true_dynamic_ffp_attention(tf.constant(ff),tf.constant(valid,dtype=tf.float32)).numpy()
    assert np.all(np.isfinite(w)) and np.all(w>=0)
    error=float(np.max(abs(w.sum(axis=(2,3))-1.)));maximum=max(maximum,error)
    assert error<2e-6
    invalid_mass+=int(np.count_nonzero(w*~valid[...,0][:,None]))
    times+=w.shape[0]*w.shape[1]
    if start%4096==0:print('Checked weight records:',start+len(rows),'/',len(idx),flush=True)
assert invalid_mass==0
report={'retained_records':len(idx),'all_timesteps':times,'unit_mass_max_abs_error':maximum,'nonzero_invalid_pixel_weights':invalid_mass,'all_finite_nonnegative':True,'all_cases_checked':True}
(ROOT/'audit/weight_full_data_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
