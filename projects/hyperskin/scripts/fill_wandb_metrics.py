import wandb
import pandas as pd
import argparse
import sys
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_ENTITY = "k298976-unicamp" 
DEFAULT_PROJECT = "hypersynth"
PREVALENCE = 0.674418605 

def compute_specificity(precision, recall, prevalence=PREVALENCE):
    """Computes specificity from precision, recall, and prevalence."""
    if precision is None or recall is None:
        return None
    if precision <= 0 or precision > 1 or recall < 0 or recall > 1:
        return None
    try:
        numerator = recall * prevalence * (1 - precision) / precision
        denominator = 1 - prevalence
        return 1 - (numerator / denominator)
    except ZeroDivisionError:
        return None

def get_metrics_at_best_f1(run):
    print(f"  -> Processing {run.name} ({run.id})...")
    
    # 1. Fetch FULL history (no keys filter) to ensure we see everything
    #    We increase samples to 100,000 to cover long training runs
    try:
        history = run.history(samples=100000, pandas=True)
    except Exception as e:
        print(f"    ⚠️ Error fetching history: {e}")
        return None

    if history.empty:
        print("    ⚠️ History is empty.")
        return None

    # 2. Determine which metric to maximize
    #    If 'val/f1_best' exists in history, use it. 
    #    Otherwise, fall back to 'val/f1' (standard practice).
    target_metric = "val/f1"
    
    if "val/f1_best" in history.columns:
        target_metric = "val/f1_best"
    elif "val/f1" in history.columns:
        target_metric = "val/f1"
    else:
        print(f"    ⚠️ Could not find 'val/f1' or 'val/f1_best' in history.")
        print(f"       Available columns: {list(history.columns)}")
        return None

    # 3. Find the index (row) with the Maximum value
    best_idx = history[target_metric].idxmax()
    
    if pd.isna(best_idx):
        return None

    best_row = history.loc[best_idx]
    
    # 4. Extract Metrics from that specific row
    f1 = best_row.get(target_metric)
    
    # Check for Accuracy variations
    accuracy = best_row.get("val/acc")
    if pd.isna(accuracy):
        accuracy = best_row.get("val/accuracy")

    prec = best_row.get("val/prec")
    rec = best_row.get("val/rec")
    spec_at_sens = best_row.get("val/spec@sens=0.95")

    # 5. Compute Specificity & Balanced Accuracy
    specificity = best_row.get("val/specificity")
    if pd.isna(specificity) or specificity is None:
        specificity = compute_specificity(prec, rec, PREVALENCE)

    bacc = best_row.get("val/bacc")
    if (pd.isna(bacc) or bacc is None) and rec is not None and specificity is not None:
        bacc = (rec + specificity) / 2

    print(f"     [Found Best] Step: {best_row.get('_step')} | {target_metric}: {f1:.4f}")

    return {
        "Wandb Name": run.name,
        "WandbID": run.id,
        "Wandb Link": run.url,
        "#": int(best_row.get("_step", 0)),
        "F1": f1,
        "Accuracy": accuracy,
        "Prec": prec,
        "Rec": rec,
        "Specificity": specificity,
        "Balanced Accuracy": bacc,
        "SpecAtSens": spec_at_sens
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wandb_id_prefix", type=str, help="First part of WandB ID (e.g. moskbpsm)")
    parser.add_argument("--entity", type=str, default=DEFAULT_ENTITY)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=str, default="filled_metrics.csv")
    args = parser.parse_args()

    print(f"🔌 Connecting to {args.entity}/{args.project}...")
    api = wandb.Api()
    
    # Fetch all runs first (most robust method)
    all_runs = api.runs(f"{args.entity}/{args.project}")
    
    print(f"🔎 Scanning {len(all_runs)} runs for ID containing '{args.wandb_id_prefix}'...")
    
    matched_runs = []
    for run in all_runs:
        if args.wandb_id_prefix in run.id:
            matched_runs.append(run)

    if not matched_runs:
        print(f"❌ No runs found with ID containing '{args.wandb_id_prefix}'.")
        sys.exit(1)

    print(f"✅ Found {len(matched_runs)} matching runs. Processing...")

    results = []
    for run in matched_runs:
        metrics = get_metrics_at_best_f1(run)
        if metrics:
            results.append(metrics)

    if not results:
        print("⚠️  No metrics extracted.")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Clean Columns
    column_order = [
        "Wandb Name", "WandbID", "Wandb Link", "#", "F1", 
        "Accuracy", "Prec", "Rec", "Specificity", 
        "Balanced Accuracy", "SpecAtSens"
    ]
    final_cols = [c for c in column_order if c in df.columns]
    df = df[final_cols]
    
    # Sort
    df = df.sort_values(by="WandbID")

    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("\n")
    output = args.output.split('.')[0] + '_' + args.wandb_id_prefix + '.csv' 
    df.to_csv(output, index=False, sep=',', float_format='%.4f')
    print(f"💾 Saved to {args.output}")

if __name__ == "__main__":
    main()