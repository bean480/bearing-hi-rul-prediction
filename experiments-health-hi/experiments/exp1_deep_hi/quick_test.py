"""Quick test: 1 fold, configurable epochs, evaluate + compare with baseline."""
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
        s = np.convolve(HI, np.ones(SG_WINDOW)/SG_WINDOW, mode='same')
        raw = s.var()/((HI-s).var()+1e-10); snr_val = raw/(1+raw)
    es = HI[int(0.2*T)]-HI[int(0.05*T)]
    return {'mon':mon,'trend':trend,'rob':rob,'snr':snr_val,'es':es,
            'comp':MON_WEIGHT*mon+TREND_WEIGHT*trend+ROB_WEIGHT*rob+SNR_WEIGHT*snr_val}

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--condition', type=int, default=1)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=== Quick Test: Cond{args.condition} Fold0, {args.epochs} epochs | {device} ===", flush=True)
    print("[1] Loading...", flush=True)
    all_ds = load_all_bearings()
    b = BEARINGS_BY_CONDITION[args.condition]
    test_name = b[0]; train_names = b[1:]
    print(f"     Train: {train_names}  Test: {test_name}")

    print(f"[2] Training {args.epochs} epochs...", flush=True)
    t0 = time.time()
    train_ds = {n: all_ds[n] for n in train_names}
    model = train_model(train_ds, epochs=args.epochs, ablation=None, device=device, verbose=True)
    print(f"     Time: {time.time()-t0:.1f}s")

    print("[3] Testing + Evaluation...", flush=True)
    model.eval()
    test_ds = all_ds[test_name]
    data = test_ds.to_torch(device)
    with torch.no_grad():
        HI = model.extract_hi(data['windows']).cpu().numpy()

    s = evaluate_hi(HI)
    print(f"\n=== Results ===")
    print(f" HI range:       [{HI.min():.4f}, {HI.max():.4f}]")
    print(f" Monotonicity:   {s['mon']:.4f}")
    print(f" Trendability:   {s['trend']:.4f}")
    print(f" Robustness:     {s['rob']:.4f}")
    print(f" SNR:            {s['snr']:.4f}")
    print(f" Early Sens:     {s['es']:.4f}")
    print(f" Composite:      {s['comp']:.4f}")
