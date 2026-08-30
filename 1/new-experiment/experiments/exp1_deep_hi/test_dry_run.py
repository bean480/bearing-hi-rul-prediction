import sys, time, numpy as np, torch
sys.path.insert(0, '.')
from data_loader import load_all_bearings
from config import BEARINGS_BY_CONDITION, MON_WEIGHT, TREND_WEIGHT, ROB_WEIGHT, SNR_WEIGHT, SG_WINDOW
from train import train_model

def _pearson(x, y):
    xc, yc = x-x.mean(), y-y.mean()
    return (xc*yc).mean() / (xc.std()*yc.std() + 1e-10)

def evaluate_hi(HI):
    HI = np.asarray(HI, dtype=np.float64); T = len(HI)
    mon = np.mean(np.diff(HI)>0) if T>1 else 0.0
    t = np.linspace(0,1,T)
    trend = max(0.0, (_pearson(HI, t)+1)/2)
    diff = np.abs(np.diff(HI))
    rob = 1.0 if diff.mean()<1e-10 else np.exp(-diff.std()/diff.mean())
    snr_val = 0.0
    if T>=SG_WINDOW:
        s = np.convolve(HI,np.ones(SG_WINDOW)/SG_WINDOW,mode='same')
        raw = s.var()/((HI-s).var()+1e-10); snr_val = raw/(1+raw)
    es = HI[int(0.2*T)]-HI[int(0.05*T)]
    return {'mon':mon,'trend':trend,'rob':rob,'snr':snr_val,'es':es,
            'comp':MON_WEIGHT*mon+TREND_WEIGHT*trend+ROB_WEIGHT*rob+SNR_WEIGHT*snr_val}

print("=== Dry Run (Cond1 Fold0, 10 epochs) ===")
print("[1] Loading...", flush=True)
all_ds = load_all_bearings()
b = BEARINGS_BY_CONDITION[1]
test_name = b[0]; train_names = b[1:]
print(f"     Train: {train_names}  Test: {test_name}")

print("[2] Training...", flush=True)
t0 = time.time()
train_ds = {n: all_ds[n] for n in train_names}
model = train_model(train_ds, epochs=10, ablation=None, device='cpu', verbose=False)
print(f"     Time: {time.time()-t0:.1f}s")

print("[3] Testing...", flush=True)
model.eval()
test_ds = all_ds[test_name]
data = test_ds.to_torch('cpu')
with torch.no_grad():
    HI = model.extract_hi(data['windows']).cpu().numpy()

scores = evaluate_hi(HI)
print(f"     HI: [{HI.min():.3f}, {HI.max():.3f}]")
print(f"     Mon={scores['mon']:.4f}  Trend={scores['trend']:.4f}  Rob={scores['rob']:.4f}")
print(f"     SNR={scores['snr']:.4f}  ES={scores['es']:.4f}")
print(f"     Composite={scores['comp']:.4f}")

print("[4] Manual SQP comparison...", flush=True)
try:
    import pandas as pd
    df = pd.read_csv("c:/Users/53031/Desktop/new-exp/results/exp1/all_bearings_feature_evaluation.csv")
    row = df[df.iloc[:,0].astype(str).str.contains(test_name, na=False)]
    if len(row)>0:
        mc = row.iloc[0]['composite_score']
        print(f"     Manual SQP: {mc:.4f}")
        print(f"     Deep HI:    {scores['comp']:.4f}")
        print(f"     Delta:      {scores['comp']-mc:+.4f}")
except Exception as e:
    print(f"     (skipped: {e})")

print("\n=== Dry Run PASSED ===")
