"""实验三: 四维特征筛选 (Mon ∩ Trend ∩ Rob ∩ Phys)"""
import sys, os, numpy as np, pandas as pd, json, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../exp_feature_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../exp_vmd_test'))

fs=25600;BPFO=(8/2)*35.0*(1-7.92/34.55)
BPFI=(8/2)*35.0*(1+7.92/34.55)

# ========== Load features ==========
pool_A=np.load('../exp_feature_pool/features/pool_A.npz')
pool_B=np.load('../exp_feature_pool/features/pool_B.npz')
with open('../exp_feature_pool/features/meta.json') as f: meta=json.load(f)

names_A=sorted(pool_A.keys());names_B=sorted(pool_B.keys())
all_features={**{f'A_{n}':pool_A[n] for n in names_A},
              **{f'B_{n}':pool_B[n] for n in names_B}}
all_names=list(all_features.keys())
n_csv=meta['n_csv'];t_vals=np.linspace(0,1,n_csv)
print(f'Loaded: {len(all_features)} features ({len(names_A)} A + {len(names_B)} B) × {n_csv} CSVs')

# ========== Compute BPFO envelope energy (for Phys) ==========
csv_files=sorted(glob.glob('../../data/XJTU-SY/Bearing1_1/*.csv'),
                 key=lambda x:int(x.split('/')[-1].split('.')[0]))

# Load or compute BPFO energy per CSV
bpfo_path='bpfo_energy.npy'
if os.path.exists(bpfo_path):
    bpfo_energy=np.load(bpfo_path)
    print(f'Loaded precomputed BPFO energy')
else:
    bpfo_energy=np.zeros(n_csv)
    for ci,cp in enumerate(csv_files):
        df=pd.read_csv(cp,header=0);sig=df.iloc[:,0].values.astype(np.float64);sig=sig-np.mean(sig)
        env=np.abs(hilbert(sig));env=env-np.mean(env)
        ef=np.abs(fft(env))[:len(sig)//2];freqs=fftfreq(len(sig),1/fs)[:len(sig)//2]
        m=(freqs>=BPFO-10)&(freqs<=BPFO+10)
        bpfo_energy[ci]=ef[m].sum() if m.sum()>0 else 0
    np.save(bpfo_path,bpfo_energy)
    print(f'Computed & saved BPFO energy')

bpfo_norm=(bpfo_energy-bpfo_energy.min())/(bpfo_energy.max()-bpfo_energy.min()+1e-10)

# ========== Compute 4 metrics ==========
def mon_score(feature):
    if len(feature)<2: return 0.0
    return np.mean(np.diff(feature)>0)

def trend_score(feature):
    if len(feature)<2: return 0.0
    r,_=pearsonr(feature,t_vals)
    return (r+1)/2

def rob_score(feature):
    if len(feature)<2: return 0.0
    d=np.abs(np.diff(feature))
    return np.exp(-d.std()/(d.mean()+1e-10))

def phys_score(feature):
    r,_=pearsonr(feature,bpfo_norm)
    return abs(r)

results={}
for name,vals in all_features.items():
    results[name]={
        'Mon':mon_score(vals),'Trend':trend_score(vals),
        'Rob':rob_score(vals),'Phys':phys_score(vals),
    }

# ========== Print scores ==========
print(f'\n{"Feature":<22} {"Mon":>6} {"Trend":>6} {"Rob":>6} {"Phys":>6}')
print('-'*50)
for name in all_names:
    s=results[name]
    print(f'{name:<22} {s["Mon"]:6.3f} {s["Trend"]:6.3f} {s["Rob"]:6.3f} {s["Phys"]:6.3f}')

# ========== Intersection pre-selection ==========
M=15  # top-M per metric
metrics=['Mon','Trend','Rob','Phys']
top_sets={}
for m in metrics:
    sorted_names=sorted(all_names,key=lambda n:results[n][m],reverse=True)
    top_sets[m]=set(sorted_names[:M])
    print(f'\nTop {M} by {m}: {sorted_names[:5]}...')

intersection=top_sets['Mon'] & top_sets['Trend'] & top_sets['Rob'] & top_sets['Phys']
print(f'\nIntersection (Mon ∩ Trend ∩ Rob ∩ Phys): {len(intersection)} features')
for n in sorted(intersection): print(f'  {n}')

# ========== Clustering re-selection ==========
if len(intersection)>12:
    from scipy.cluster.hierarchy import linkage,fcluster
    # Build feature matrix for intersection only
    inter_list=sorted(intersection)
    feat_mat=np.array([all_features[n] for n in inter_list])  # [N, n_csv]
    # Correlation distance
    corr_mat=np.corrcoef(feat_mat)
    Z=linkage(1-np.abs(corr_mat),method='average')
    n_clusters=min(8,len(inter_list)//2)
    clusters=fcluster(Z,n_clusters,criterion='maxclust')
    # Pick closest to cluster center
    final=[]
    print(f'\nClustering: {n_clusters} clusters from {len(inter_list)} features')
    for c in range(1,n_clusters+1):
        idx=np.where(clusters==c)[0]
        center=feat_mat[idx].mean(axis=0)
        dists=np.linalg.norm(feat_mat[idx]-center,axis=1)
        best=idx[np.argmin(dists)]
        final.append(inter_list[best])
        others=[inter_list[i] for i in idx if i!=best]
        print(f'  Cluster {c}: best={inter_list[best]}, others={others}')
    final_selected=sorted(final)
else:
    final_selected=sorted(intersection)
    print(f'\nNo clustering needed ({len(intersection)} features)')

print(f'\nFinal selected: {len(final_selected)} features')
for n in final_selected: print(f'  {n}')

# ========== FIGURE ==========
fig=plt.figure(figsize=(20,12))
gs=fig.add_gridspec(2,3,hspace=0.35,wspace=0.3)

# Row 0: 4D bar chart (one bar chart per metric, colored by selected)
for col,(metric,title) in enumerate([('Mon','Monotonicity ↑'),('Trend','Trendability ↑'),('Rob','Robustness ↑'),('Phys','Physical Fidelity ↑')]):
    ax=fig.add_subplot(gs[0,col])
    names_sorted=sorted(all_names,key=lambda n:results[n][metric],reverse=True)
    vals=[results[n][metric] for n in names_sorted]
    colors=['#C44E52' if n in set(final_selected) else 'steelblue' for n in names_sorted]
    ax.bar(range(len(vals)),vals,color=colors,edgecolor='none',width=0.8)
    # Mark intersection cutoff
    ax.axvline(M-0.5,color='gray',ls='--',lw=1.5)
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels([n.split('_')[0]+'_'+n.split('_')[1][:4] for n in names_sorted],fontsize=5,rotation=90)
    ax.set_ylabel(metric);ax.set_title(title,fontsize=11,fontweight='bold')
    ax.set_ylim(0,1.05);ax.grid(True,alpha=0.2,axis='y')

# Row 1 col 0-1: Radar chart of final selected features
ax=fig.add_subplot(gs[1,0])
angles=np.linspace(0,2*np.pi,4,endpoint=False).tolist()
angles+=angles[:1]
for name in final_selected:
    s=results[name]
    vals=[s['Mon'],s['Trend'],s['Rob'],s['Phys']]
    vals+=vals[:1]
    ax.plot(angles,vals,'o-',lw=1.2,alpha=0.7,label=name[:25],markersize=3)
ax.set_xticks(angles[:-1]);ax.set_xticklabels(metrics,fontsize=10)
ax.set_ylim(0,1);ax.set_title(f'Selected Features ({len(final_selected)})',fontsize=12,fontweight='bold')
ax.legend(fontsize=6,loc='lower right',ncol=2)
ax.grid(True,alpha=0.2)

# Row 1 col 1: Intersection Venn-style (counts)
ax=fig.add_subplot(gs[1,1])
# Show how many features pass each dimension
pass_counts=[len(top_sets[m]) for m in metrics]
intersection_counts=[]
for i,m in enumerate(metrics):
    if i==0: inter=top_sets[m]
    else: inter=inter & top_sets[m]
    intersection_counts.append(len(inter))
ax.bar(['Each Top15']+metrics,[45]+pass_counts,color='steelblue')
ax.bar(metrics+'+Intersection',intersection_counts,color='#C44E52')
ax.set_ylabel('Count');ax.set_title('Feature Count by Dimension',fontsize=12,fontweight='bold')
ax.grid(True,alpha=0.2,axis='y')

# Row 1 col 2: Summary table
ax=fig.add_subplot(gs[1,2])
ax.axis('off')
lines=[f'Feature Screening Results','','Bearing1_1 ({n_csv} CSVs)','',
       f'Total features: {len(all_features)}',
       f'  Pool A: {len(names_A)} (VMD Recon)',
       f'  Pool B: {len(names_B)} (Original)','',
       f'Pre-selection (M={M}):',
       f'  Mon ∩ Trend ∩ Rob ∩ Phys',
       f'  = {len(intersection)} features','',
       f'Final: {len(final_selected)} features']
max_len=max(len(l) for l in lines)
for i,line in enumerate(lines):
    ax.text(0.05,0.95-i*0.06,line,fontsize=9 if i>0 else 10,va='top',fontweight='bold' if 'Final' in line else 'normal')

fig.suptitle('Four-Dimensional Feature Screening | Bearing1_1\n'
             f'Mon ∩ Trend ∩ Rob ∩ Phys | {len(final_selected)}/{len(all_features)} features selected',
             fontsize=14,fontweight='bold',y=1.01)
plt.tight_layout()
fig.savefig('figures/01_feature_screening.png',dpi=150,bbox_inches='tight')
plt.close(fig)

# Save
np.savez('selected_features.npz',
         names=np.array(final_selected),
         features=np.array([all_features[n] for n in final_selected]))
print(f'\nSaved: selected_features.npz, figures/01_feature_screening.png')

import time;os.utime('figures/01_feature_screening.png',(time.time(),time.time()))
