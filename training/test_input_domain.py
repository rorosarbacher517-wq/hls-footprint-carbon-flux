from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parent
inp=Path(json.loads((ROOT/'config.json').read_text())['input_root'])
h=np.load(inp/'HLS_matched_v11_remove_selected_sites.npy',mmap_mode='r')
counts=np.zeros(6,np.int64)
for start in range(0,len(h),256):counts+=np.sum(~np.isfinite(h[start:start+256,:6]),axis=(0,2,3))
assert not counts.any(),'Original normalizer cannot accept nonfinite source HLS values'
report={'all_source_records':len(h),'used_HLS_band_nonfinite_counts':counts.tolist(),'original_normalizer_finite_input_requirement_passed':True}
(ROOT/'audit/HLS_input_domain_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
