from pathlib import Path
import sys,os,json,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'FAT'))
from numpy_pickle_compat import install_numpy_pickle_compat
install_numpy_pickle_compat()
from verify import verify,sha
verify()
config=json.loads((ROOT/'config.json').read_text());inp=Path(config['input_root'])
manifest={}
for name in ['HLS_matched_v11_remove_selected_sites.npy','FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy','merged_flux_era5_ffp_array_v11_remove_selected_sites.npy']:
    print('Hashing '+name,flush=True);p=inp/name;manifest[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
(ROOT/'reference/input_manifest.json').write_text(json.dumps(manifest,indent=2))
from audit_height_uniform import audit
height=audit(inp,ROOT/'reference/split_kfold',ROOT/'audit/uniform_windows.csv')
assert height['fold_mismatch_count']==0 and height['records_without_height']==0 and height['retained_records_without_support']==0
assert height['max_abs_zm_minus_MH_minus_d']<.001
(ROOT/'audit/height_check.json').write_text(json.dumps(height,indent=2))
print('Height and frozen split checks passed',flush=True)
flux=np.load(inp/'merged_flux_era5_ffp_array_v11_remove_selected_sites.npy',allow_pickle=True)
hls=np.load(inp/'HLS_matched_v11_remove_selected_sites.npy',mmap_mode='r')
y=np.asarray(flux[:,:,8:11],np.float32)
valid=np.isfinite(y)&(y!=-9999.)
assert np.all(np.isfinite(y))
assert np.array_equal(valid[...,0],valid[...,1]) and np.array_equal(valid[...,0],valid[...,2]),'RF joint label mask mismatch'
keep=np.all(hls[:,0,23:27,23:27]!=-9999.,axis=(1,2))
sensor=np.asarray(flux[keep,:,6],np.float32)
assert np.all(np.isfinite(sensor)&((sensor==0)|(sensor==1))), 'Invalid raw Sensor category'
print('Raw Sensor: all retained input time slots are valid 0/1 categories.',flush=True)
from fair_interfaces import COVARIATES,digest_array
x=np.asarray(flux[:,:,[i+1 for i in COVARIATES]],np.float32)
assert not np.any(valid[...,0]&keep[:,None]&~np.all(np.isfinite(x)&(x!=-9999.),axis=-1)),'RF feature filtering would drop labels'
report={'height':height,'folds':[],'raw_sensor_0_1_all_retained_slots':True,'raw_label_masks_equal':True,'labeled_covariates_valid':True}
for k in range(5):
    tr=np.flatnonzero(np.load(ROOT/'reference/split_kfold'/('index_train_fold%d.npy'%k))&keep)
    te=np.flatnonzero(np.load(ROOT/'reference/split_kfold'/('index_test_fold%d.npy'%k))&keep)
    cut=int(len(tr)*.96)
    assert len(np.intersect1d(tr,te))==0
    report['folds'].append({'fold':k,'outer_train_retained':len(tr),'fit':cut,'validation':len(tr)-cut,'test':len(te),'test_valid_n':valid[te].sum(axis=(0,1)).tolist(),'fit_sha256':digest_array(tr[:cut]),'validation_sha256':digest_array(tr[cut:]),'test_sha256':digest_array(te)})
    np.savez(ROOT/'reference'/('fold%d_effective_indices.npz'%k),train=tr[:cut],validation=tr[cut:],test=te,outer_train_retained=tr)
report['pooled_test_n']=[sum(f['test_valid_n'][i] for f in report['folds']) for i in range(3)]
(ROOT/'audit/data_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
