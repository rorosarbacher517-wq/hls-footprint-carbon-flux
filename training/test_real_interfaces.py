import os,sys,json,csv,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'FAT'))
import numpy as np
from numpy_pickle_compat import install_numpy_pickle_compat
install_numpy_pickle_compat()
from fair_interfaces import aggregate_reference,validate_predictions,host_inputs
from Model_build import preprocess_inputs
c=json.loads((ROOT/'config.json').read_text());inp=Path(c['input_root'])
hls=np.load(inp/'HLS_matched_v11_remove_selected_sites.npy',mmap_mode='r');ffp=np.load(inp/'FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy',mmap_mode='r')
idx=np.load(ROOT/'reference/fold0_effective_indices.npz')['test'];rows=idx[np.linspace(0,len(idx)-1,32,dtype=int)];rows=np.unique(np.r_[rows,7122])
# Historical train-only stats are used ONLY to make realistic interface test values.
# Production always recomputes stats with the original FAT normalizer.
stats=ROOT/'reference/interface_test_Joint_mean_std_fold0.csv'
with stats.open() as f:table=list(csv.DictReader(f))
mean=np.array([float(x['mean']) for x in table[:6]],np.float32);std=np.array([float(x['std']) for x in table[:6]],np.float32)
h=np.transpose(np.array(hls[rows,:6]),(0,2,3,1));h=np.where(h!=-9999.,(h-mean)/std,-9999.).astype(np.float32);f=np.array(ffp[rows])
flux=np.zeros((len(rows),48,110),object);flux[:,:,0]='synthetic_site'
for j in range(1,110):flux[:,:,j]=np.float32(j/100.)
flux[:,:,6]=0.;flux[:,:,8:11]=1.
sites,ht,ft,wt=host_inputs(preprocess_inputs,h,flux,f)
np.testing.assert_array_equal(ht.numpy(),h);np.testing.assert_array_equal(wt.numpy(),f);np.testing.assert_array_equal(ft.numpy(),flux[:,:,1:].astype(np.float32))
expected=aggregate_reference(ht,wt)
spec=importlib.util.spec_from_file_location('audited_rf',ROOT/'RF/Pro_RF_HLS_carbon_auto_merge3_stable_pred_layout_fixed_fairfix_remove_selected_sites.py');rf=importlib.util.module_from_spec(spec);spec.loader.exec_module(rf)
x,y,mask,info=rf.build_rf_features_common(ht,ft,wt,'dynamic_ffp',auto_drop_bad_flux_cols=False,debug=False)
np.testing.assert_array_equal(x[:,:6],expected.reshape(-1,6))
assert mask.all() and info['selected_flux_feature_indices']==rf.get_fair_rf_flux_feature_indices()
try:validate_predictions(np.ones((1,1,3)),np.full((1,1,3),np.nan))
except RuntimeError:pass
else:raise AssertionError('Nonfinite prediction guard failed')
report={'real_HLS_FFP_records':len(rows),'CPU_staging_values_exact':True,'RF_and_PREAGG_HLS_features_exact':True,'RF_feature_count':int(x.shape[1]),'RF_covariate_columns':info['selected_flux_feature_indices'],'covariates_for_this_interface_test':'synthetic finite values; full real covariate validity separately audited','nonfinite_prediction_guard_passed':True,'training_performed':False}
(ROOT/'audit/real_interface_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
