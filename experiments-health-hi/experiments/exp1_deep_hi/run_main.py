"""
实验1 深度HI — 完整实验主脚本
训练 + 测试 + 评估 + 对比 + 可视化
"""

import sys, os, time, json, numpy as np, torch
from datetime import datetime
sys.path.insert(0, '.')
from config import (OUTPUT_ROOT, BEARINGS_BY_CONDITION, EPOCHS,
                     MON_WEIGHT, TREND_WEIGHT, ROB_WEIGHT, SNR_WEIGHT, SG_WINDOW)
from data_loader import load_all_bearings
from train import train_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RUN_ABLATION = False  # 跑消融实验? 先跑主体的用 False


# ============================================================
# 评估函数 (内嵌)
# ============================================================

def _smooth(HI, window=11):
    if len(HI) < window: return HI.copy()
    return np.convolve(HI, np.ones(window)/window, mode='same')

def evaluate_hi(HI):
    HI = np.asarray(HI, dtype=np.float64)
    T = len(HI)
    mon = np.mean(np.diff(HI) > 0) if T > 1 else 0.0
    t = np.linspace(0, 1, T)
    c = ((HI-HI.mean())*(t-t.mean())).mean()/((HI-HI.mean()).std()*(t-t.mean()).std()+1e-10)
    trend = 0.0 if np.isnan(c) else (c + 1) / 2
    diff = np.abs(np.diff(HI))
    rob = 1.0 if diff.mean() < 1e-10 else np.exp(-diff.std() / diff.mean())
    snr_val = 0.0
    if T >= SG_WINDOW:
        smooth = _smooth(HI, SG_WINDOW)
        noise = HI - smooth
        raw = smooth.var() / (noise.var() + 1e-10)
        snr_val = raw / (1 + raw)
    es = HI[int(0.20*T)] - HI[int(0.05*T)]
    comp = MON_WEIGHT*mon + TREND_WEIGHT*trend + ROB_WEIGHT*rob + SNR_WEIGHT*snr_val
    return {'monotonicity':mon,'trendability':trend,'robustness':rob,
            'snr':snr_val,'early_sensitivity':es,'composite':comp}


# ============================================================
# 日志
# ============================================================

def log(msg):
    print(msg)
    with open(f"{OUTPUT_ROOT}/experiment.log", 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ============================================================
# 主流程
# ============================================================

def main():
    t_start = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/hi_data", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/figures", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/checkpoints", exist_ok=True)

    log("="*60)
    log(f"Experiment Start: {datetime.now()}")
    log(f"Device: {DEVICE}, Epochs per fold: {EPOCHS}")
    log("="*60)

    # ---- Load data ----
    log("\n[1/4] Loading data...")
    all_ds = load_all_bearings()
    total_w = sum(d.T for d in all_ds.values())
    log(f"  {len(all_ds)} bearings, {total_w} windows")

    # ---- Load manual baseline ----
    log("\n[2/4] Loading manual baseline (SQP method)...")
    manual_scores = {}
    try:
        import pandas as pd
        manual_csv = f"{OUTPUT_ROOT}/../exp1/all_bearings_feature_evaluation.csv"
        if os.path.exists(manual_csv):
            df = pd.read_csv(manual_csv)
            for _, row in df.iterrows():
                manual_scores[row.get('bearing','')] = {
                    'monotonicity': row.get('monotonicity', 0),
                    'trendability': row.get('trendability', 0),
                    'robustness': row.get('robustness', 0),
                    'composite': row.get('composite_score', 0),
                }
            log(f"  Found manual baseline for {len(manual_scores)} bearings")
        else:
            log("  No manual baseline found, skipping comparison")
    except Exception as e:
        log(f"  Baseline load error: {e}")

    # ---- Leave-one-out CV per condition ----
    log("\n[3/4] Leave-one-out CV (within condition)...")
    all_results = {}  # {bearing_name: {HI, scores, ...}}
    all_summaries = {}

    for condition in [1, 2, 3]:
        bearings = BEARINGS_BY_CONDITION[condition]
        log(f"\n{'='*50}")
        log(f"Condition {condition}: {bearings}")
        log(f"{'='*50}")

        cond_scores = []

        for test_name in bearings:
            train_names = [b for b in bearings if b != test_name]
            log(f"\n  Train: {train_names}")
            log(f"  Test:  {test_name}")

            t_fold = time.time()

            # Train
            train_ds = {n: all_ds[n] for n in train_names}
            model = train_model(train_ds, epochs=EPOCHS, ablation=None, device=DEVICE, verbose=True)

            # Evaluate on test bearing
            model.eval()
            test_ds = all_ds[test_name]
            data = test_ds.to_torch(DEVICE)
            with torch.no_grad():
                HI = model.extract_hi(data['windows']).cpu().numpy()

            scores = evaluate_hi(HI)
            cond_scores.append(scores)

            log(f"  Result: Mon={scores['monotonicity']:.4f} Trend={scores['trendability']:.4f} "
                f"Rob={scores['robustness']:.4f} SNR={scores['snr']:.4f} "
                f"Comp={scores['composite']:.4f}")

            # Comparison with manual method
            if test_name in manual_scores:
                ms = manual_scores[test_name]
                delta = scores['composite'] - ms.get('composite', 0)
                log(f"  Manual SQP: Comp={ms.get('composite',0):.4f}  Delta={delta:+.4f}")

            # Save
            all_results[test_name] = {'HI': HI, 'scores': scores, 'condition': condition}
            np.savez(f"{OUTPUT_ROOT}/hi_data/{test_name}.npz", HI=HI)
            torch.save(model.state_dict(), f"{OUTPUT_ROOT}/checkpoints/{test_name}.pt")
            log(f"  Time: {time.time()-t_fold:.1f}s")

        # Condition summary
        keys = ['monotonicity','trendability','robustness','snr','early_sensitivity','composite']
        mean_s = {k: np.mean([s[k] for s in cond_scores]) for k in keys}
        std_s = {k: np.std([s[k] for s in cond_scores]) for k in keys}
        all_summaries[condition] = {'mean': mean_s, 'std': std_s}

        log(f"\n  Condition {condition} Summary:")
        log(f"  Comp: {mean_s['composite']:.4f} +/- {std_s['composite']:.4f}")
        log(f"  Mon:  {mean_s['monotonicity']:.4f} +/- {std_s['monotonicity']:.4f}")
        log(f"  Trend:{mean_s['trendability']:.4f} +/- {std_s['trendability']:.4f}")

    # ---- Overall summary ----
    log(f"\n{'='*60}")
    log(f"[4/4] Overall Results")
    log(f"{'='*60}")

    all_keys = ['monotonicity','trendability','robustness','snr','early_sensitivity','composite']
    overall_mean = {}
    overall_std = {}
    for k in all_keys:
        values = [all_results[n]['scores'][k] for n in all_results]
        overall_mean[k] = np.mean(values)
        overall_std[k] = np.std(values)
        log(f"  {k:<20}: {overall_mean[k]:.4f} +/- {overall_std[k]:.4f}")

    # Deep vs Manual comparison
    if manual_scores:
        log(f"\n{'='*60}")
        log(f"Deep HI vs Manual SQP")
        log(f"{'='*60}")
        deep_comp = [all_results[n]['scores']['composite'] for n in all_results if n in manual_scores]
        manual_comp = [manual_scores[n]['composite'] for n in all_results if n in manual_scores]
        log(f"  Deep:   {np.mean(deep_comp):.4f} +/- {np.std(deep_comp):.4f}")
        log(f"  Manual: {np.mean(manual_comp):.4f} +/- {np.std(manual_comp):.4f}")
        log(f"  Delta:  {np.mean(deep_comp)-np.mean(manual_comp):+.4f}")

    # Save summary
    json.dump({'per_bearing': {n: all_results[n]['scores'] for n in all_results},
               'overall_mean': overall_mean, 'overall_std': overall_std,
               'condition_summaries': {str(c): {'mean':all_summaries[c]['mean'],
                                                 'std':all_summaries[c]['std']} for c in [1,2,3]}},
              open(f"{OUTPUT_ROOT}/summary.json", 'w'), indent=2)

    log(f"\n{'='*60}")
    log(f"Experiment Complete! Total: {(time.time()-t_start)/60:.1f} min")
    log(f"Results: {OUTPUT_ROOT}/")
    log(f"{'='*60}")


if __name__ == '__main__':
    main()
