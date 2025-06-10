import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# File paths for each policy
RESULT_FILES = {
    'GRU': 'test_results_finbert_gru_standalone.csv',
    'Transformer': 'test_results_finbert_transformer_standalone.csv',
    'MLP': 'test_results_finbert.csv',
}

def load_policy_results():
    results = {}
    for policy, path in RESULT_FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            results[policy] = df
        else:
            print(f"Warning: {path} not found.")
    return results

def summarize_policy(df):
    return {
        'avg_return': df['return_pct'].mean(),
        'avg_drawdown': df['max_drawdown'].mean(),
        'avg_sharpe': df['sharpe_ratio'].mean(),
        'max_return': df['return_pct'].max(),
        'min_return': df['return_pct'].min(),
        'n': len(df)
    }

def print_summary_table(summaries):
    print("\nSummary Table:")
    print(f"{'Policy':<14} {'AvgReturn(%)':>12} {'AvgDrawdown(%)':>15} {'AvgSharpe':>10} {'MaxReturn':>10} {'MinReturn':>10} {'N':>5}")
    for policy, stats in summaries.items():
        print(f"{policy:<14} {stats['avg_return']:12.2f} {stats['avg_drawdown']:15.2f} {stats['avg_sharpe']:10.2f} {stats['max_return']:10.2f} {stats['min_return']:10.2f} {stats['n']:5}")

def plot_boxplots(results):
    # Only plot if at least one result is available
    if not results:
        print("No results to plot.")
        return
    data = [df['return_pct'] for df in results.values()]
    labels = list(results.keys())
    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel('Return (%)')
    plt.xlabel('Policy Architecture')
    plt.title('Distribution of Returns by Policy (FinBERT Embedding)')
    plt.tight_layout()
    plt.savefig('returns_boxplot_policies_finbert.png')
    plt.show()

def main():
    policy_results = load_policy_results()
    summaries = {}
    for policy, df in policy_results.items():
        summaries[policy] = summarize_policy(df)
    print_summary_table(summaries)
    try:
        plot_boxplots(policy_results)
    except Exception as e:
        print(f"Plotting failed: {e}")
    # Save summary CSV
    pd.DataFrame(summaries).T.to_csv('summary_analysis_policies_finbert.csv')
    print("\nSaved summary to summary_analysis_policies_finbert.csv and boxplot to returns_boxplot_policies_finbert.png")

if __name__ == "__main__":
    main() 