import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Paths
RESULTS_DIR = "benchmarks/results/attention_analysis_exp"
FILE_ON = os.path.join(RESULTS_DIR, "results_all_on.csv")
FILE_OFF = os.path.join(RESULTS_DIR, "results_all_off.csv")
OUTPUT_STATS = os.path.join(RESULTS_DIR, "final_statistics.txt")

def analyze():
    # 1. Load Data
    print(f"Loading {FILE_ON}...")
    df_on = pd.read_csv(FILE_ON)
    print(f"Loading {FILE_OFF}...")
    df_off = pd.read_csv(FILE_OFF)
    
    # Merge
    df = pd.merge(df_on, df_off, on="Sample_ID", suffixes=('_on', '_off'))
    
    # Calculate Deltas (On - Off)
    df["Delta_Perf"] = df["Performance_on"] - df["Performance_off"]
    df["Delta_MDI"] = df["MDI_on"] - df["MDI_off"]
    df["Delta_AEI"] = df["AEI_on"] - df["AEI_off"]
    
    # === Task 1: Statistics ===
    stats_output = []
    stats_output.append("=== (1) Statistics (Mean ± Variance / Median / P90) ===")
    
    metrics = ["MDI", "AEI"]
    modes = ["on", "off"]
    
    for metric in metrics:
        for mode in modes:
            col = f"{metric}_{mode}"
            data = df[col]
            mean_val = data.mean()
            var_val = data.var()
            median_val = data.median()
            p90_val = data.quantile(0.9)
            
            line = f"{metric} ({mode.upper()}): Mean={mean_val:.4f} ± Var={var_val:.4f} / Median={median_val:.4f} / P90={p90_val:.4f}"
            stats_output.append(line)
            print(line)
            
    # === Task 2: Correlations & Scatter Plots ===
    stats_output.append("\n=== (2) Correlations (Delta Perf vs Delta Metric) ===")
    
    # Remove NaNs if any
    df_clean = df.dropna(subset=["Delta_Perf", "Delta_MDI", "Delta_AEI"])
    
    # Plot Settings
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
    
    for metric in ["MDI", "AEI"]:
        delta_col = f"Delta_{metric}"
        
        # Calculate Correlation
        pearson = stats.pearsonr(df_clean[delta_col], df_clean["Delta_Perf"])
        spearman = stats.spearmanr(df_clean[delta_col], df_clean["Delta_Perf"])
        
        res_str = f"Delta {metric} vs Delta Perf: Pearson r={pearson.statistic:.4f} (p={pearson.pvalue:.4g}), Spearman r={spearman.statistic:.4f} (p={spearman.pvalue:.4g})"
        stats_output.append(res_str)
        print(res_str)
        
        # Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df_clean[delta_col], df_clean["Delta_Perf"], alpha=0.6, edgecolors='w', linewidth=0.5)
        plt.title(f"Correlation: Delta Perf vs Delta {metric}\nPearson r={pearson.statistic:.2f}, Spearman r={spearman.statistic:.2f}")
        plt.xlabel(f"Delta {metric} (On - Off)")
        plt.ylabel("Delta Performance (On - Off)")
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(RESULTS_DIR, f"scatter_corr_delta_perf_{metric.lower()}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

    # === Task 3: Binning by CoT Length ===
    stats_output.append("\n=== (3) Binning by CoT Length ===")
    
    # Use CoT_Len from 'on' mode (off is usually 0/1)
    cot_col = "CoT_Len_on" # Actually in merge it might be CoT_Len_on
    if "CoT_Len_on" not in df.columns:
        # Fallback if names differ, check columns
        # Original CSV had 'CoT_Len'. Merge makes CoT_Len_on/off.
        cot_col = "CoT_Len_on"
        
    if df[cot_col].max() > 0:
        # Define bins
        num_bins = 5
        df["CoT_Bin"] = pd.cut(df[cot_col], bins=num_bins)
        
        # Group by bin
        bin_stats = df.groupby("CoT_Bin", observed=False)["Delta_Perf"].agg(['mean', 'count', 'std', 'sem']).reset_index()
        
        print("\nBinning Stats:")
        print(bin_stats)
        stats_output.append(bin_stats.to_string())
        
        # Bar Plot with Error Bars
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(bin_stats))
        
        plt.bar(x_pos, bin_stats['mean'], yerr=bin_stats['sem'], capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        
        plt.title("Performance Degradation (Delta Perf) by CoT Length\n(Error Bars = SEM)")
        plt.xlabel("CoT Length (Tokens)")
        plt.ylabel("Mean Delta Performance (On - Off)")
        
        # Format X-axis labels
        labels = [str(x) for x in bin_stats["CoT_Bin"]]
        plt.xticks(x_pos, labels, rotation=45)
        
        plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(RESULTS_DIR, "bar_cot_len_vs_delta_perf.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
        
        # Also Scatter Plot for CoT Length vs Delta Perf
        plt.figure(figsize=(8, 6))
        plt.scatter(df[cot_col], df["Delta_Perf"], alpha=0.5, color='orange', edgecolors='grey')
        plt.title("Scatter: CoT Length vs Delta Performance")
        plt.xlabel("CoT Length (Tokens)")
        plt.ylabel("Delta Performance (On - Off)")
        plt.axhline(0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "scatter_cot_len_vs_delta_perf.png") )
        plt.close()
    else:
        msg = "CoT Length is 0 for all samples, skipping binning."
        print(msg)
        stats_output.append(msg)

    # Save Stats
    with open(OUTPUT_STATS, "w") as f:
        f.write("\n".join(stats_output))
    print(f"\nSaved statistics to {OUTPUT_STATS}")

if __name__ == "__main__":
    analyze()
