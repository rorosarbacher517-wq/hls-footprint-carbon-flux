#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the FAT gain increase with how different the two aggregation rules are?  (negative result, kept for transparency)

Exposure measures per held-out day (from weight_difference.py, averaged over the half-hours with a valid observation):
shift_m (distance between the weight centroids), tv (total-variation distance), outside (FAT weight outside the uniform
window), logneff (log ratio of effective pixel numbers).  Compared with the gain (delta RMSE) at annual / monthly / daily scale:
Spearman rho overall and within sites (site-demeaned), plus a 3x3 grid of NDVI-heterogeneity tercile x centroid-shift tercile.
Outputs: ../outputs/weight_difference_vs_gain.csv, ../outputs/heterogeneity_x_shift_grid_annual.csv
"""
import importlib.util,numpy as np,pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('g',HERE/'gain_correlates.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
d=g.daily_gain_table()
W=np.load(g.PROJECT/'data'/'weight_difference_halfhour.npz',allow_pickle=True)
rec=pd.DataFrame({'SITE_ID':W['SITE_ID'].astype(str),'Year':W['Year'],'DoY':W['DoY'],'HLS_Hour':W['HLS_Hour']})
# valid obs masks per flux, same order
y,pf,site=g.load_model('FAT')
ex={}
for i,fl in enumerate(g.FLUXES):
    obs=y[:,:,i].astype(float);v=np.isfinite(obs)&(obs>-9000)
    wv=v.sum(1)
    for name in ['shift_m','tv','outside']:
        a=W[name].astype(float);ex[(fl,name)]=np.where(wv>0,np.nansum(np.where(v,a,0),1)/np.maximum(wv,1),np.nan)
    ex[(fl,'logneff')]=np.where(wv>0,np.nansum(np.where(v,np.log(W['neff_u'].astype(float)/W['neff_f'].astype(float)),0),1)/np.maximum(wv,1),np.nan)
rows=[]
for fl in g.FLUXES:
    t=rec.copy();t['flux']=fl
    for name in ['shift_m','tv','outside','logneff']:t[name]=ex[(fl,name)]
    rows.append(t)
E=pd.concat(rows)
d=d.merge(E,on=g.KEYS+['flux'],how='left')
print('records with exposure',d['shift_m'].notna().mean())
EXPO=['shift_m','tv','outside','logneff']
def agg(d,scale):
    # add exposure columns to aggregate via weighted mean
    keys={'year':['SITE_ID','Year'],'monthly':['SITE_ID','Year','month'],'daily':['SITE_ID','Year','DoY']}[scale]
    gg=d.assign(w=d.n_valid.astype(float),md=lambda x:x.rmse_dynamic**2*x.n_valid,ms=lambda x:x.rmse_static**2*x.n_valid)
    for c in EXPO+['NDVI_STD_DAY']:gg[c+'_w']=gg[c]*gg['w']
    s=gg.groupby(keys+['flux']).agg(w=('w','sum'),md=('md','sum'),ms=('ms','sum'),n_days=('DoY','size'),**{c:(c+'_w','sum') for c in EXPO+['NDVI_STD_DAY']}).reset_index()
    for c in EXPO+['NDVI_STD_DAY']:s[c]=s[c]/s['w']
    s['delta']=np.sqrt(s.ms/s.w)-np.sqrt(s.md/s.w)
    return s[s.n_days>=g.MIN_DAYS[{'year':'year','monthly':'monthly','daily':'daily'}[scale]]]
rows=[]
for scale in ['year','monthly','daily']:
    t=agg(d,scale)
    for fl in g.FLUXES:
        x=t[t.flux==fl].dropna(subset=['delta']+EXPO+['NDVI_STD_DAY'])
        for c in EXPO+['NDVI_STD_DAY']:
            w=x.copy()
            for col in (c,'delta'):w[col]=w[col]-w.groupby('SITE_ID')[col].transform('mean')
            rows.append({'scale':scale,'flux':fl,'measure':c,'rho_overall':spearmanr(x[c],x.delta)[0],'rho_within_site':spearmanr(w[c],w.delta)[0],'n':len(x),'n_sites':x.SITE_ID.nunique()})
pd.DataFrame(rows).to_csv(HERE.parent/'outputs'/'weight_difference_vs_gain.csv',index=False)
# 3x3 grid annual NEE/GPP
t=agg(d,'year')
grid=[]
for fl in ['NEE','GPP','RECO']:
    x=t[t.flux==fl].dropna(subset=['delta','shift_m','NDVI_STD_DAY']).copy()
    x['hq']=pd.qcut(x.NDVI_STD_DAY,3,labels=['low','mid','high']);x['sq']=pd.qcut(x.shift_m,3,labels=['low','mid','high'])
    m=x.groupby(['hq','sq'],observed=True)['delta'].agg(['mean','size']).reset_index();m.insert(0,'flux',fl);grid.append(m)
pd.concat(grid).rename(columns={'hq':'heterogeneity_tercile','sq':'shift_tercile','mean':'mean_delta_rmse','size':'n_site_years'}).to_csv(HERE.parent/'outputs'/'heterogeneity_x_shift_grid_annual.csv',index=False)
