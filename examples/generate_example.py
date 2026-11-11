"""Generate fabricated, small arrays; never overwrite existing examples."""
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent/'generated'
R.mkdir(exist_ok=False)
rng=np.random.default_rng(42);n=2
h=rng.uniform(.05,.5,(n,20,50,50)).astype('float32')
y,x=np.mgrid[:50,:50];f=np.empty((n,48,50,50),np.float32)
for i in range(n):
 for t in range(48):
  w=np.exp(-((x-(20+5*np.sin(t/48*2*np.pi)))**2+(y-25)**2)/80)
  f[i,t]=w/w.sum()
v=np.zeros((n,48,110),object)
for i in range(n):
 v[i,:,0]='SYNTH-%02d'%i;v[i,:,3]=2020;v[i,:,4]=180+i
 v[i,:,6]=i;v[i,:,7]=np.arange(48)/2
 v[i,:,9]=np.maximum(0,10*np.sin((np.arange(48)-12)/48*2*np.pi))
 v[i,:,10]=3.;v[i,:,8]=v[i,:,10]-v[i,:,9]
np.save(R/'HLS_matched_v11_remove_selected_sites.npy',h)
np.save(R/'FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy',f)
np.save(R/'merged_flux_era5_ffp_array_v11_remove_selected_sites.npy',v)
print('Synthetic examples written to',R)
