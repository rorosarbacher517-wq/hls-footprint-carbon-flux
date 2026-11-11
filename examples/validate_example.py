from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent/'generated'
h=np.load(R/'HLS_matched_v11_remove_selected_sites.npy')
f=np.load(R/'FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy')
v=np.load(R/'merged_flux_era5_ffp_array_v11_remove_selected_sites.npy',allow_pickle=True)
assert h.shape==(2,20,50,50) and f.shape==(2,48,50,50) and v.shape==(2,48,110)
assert np.allclose(f.sum(axis=(-1,-2)),1) and (f>=0).all()
assert np.allclose(v[:,:,8].astype(float),v[:,:,10].astype(float)-v[:,:,9].astype(float))
print('PASS: dimensions, sample alignment, footprint sums and synthetic carbon balance')
