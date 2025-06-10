import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# File paths
RESULT_FILES = {
    'bert': 'test_results_bert.csv',
    'finbert': 'test_results_finbert.csv',
    'longformer': 'test_results_longformer.csv',
    'longt5': 'test_results_longt5.csv',
}
BASELINE_PORTFOLIO = 'finbert_portfolio.csv'

# Helper to load RL results
def load_rl_results():
    results = {}
    for model, path in RESULT_FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            results[model] = df
        else:
            print(f"Warning: {path} not found.")
    return results

def load_baseline():
    if os.path.exists(BASELINE_PORTFOLIO):
        df = pd.read_csv(BASELINE_PORTFOLIO)
        return df
    else:
        print(f"Warning: {BASELINE_PORTFOLIO} not found.")
        return None

def summarize_rl(df):
    return {
        'avg_return': df['return_pct'].mean(),
        'avg_drawdown': df['max_drawdown'].mean(),
        'avg_sharpe': df['sharpe_ratio'].mean(),
        'max_return': df['return_pct'].max(),
        'min_return': df['return_pct'].min(),
        'n': len(df)
    }

def summarize_baseline(df):
    # Use per-period returns for average, max, min
    per_period_returns = df['returns'].dropna() * 100  # convert to percent
    avg_return = per_period_returns.mean()
    max_return = per_period_returns.max()
    min_return = per_period_returns.min()
    max_drawdown = df['drawdown'].min() * 100
    sharpe = np.sqrt(252) * (per_period_returns.mean() / per_period_returns.std()) if per_period_returns.std() > 0 else 0
    return {
        'avg_return': avg_return,
        'avg_drawdown': max_drawdown,
        'avg_sharpe': sharpe,
        'max_return': max_return,
        'min_return': min_return,
        'n': len(per_period_returns)
    }

def print_summary_table(summaries):
    print("\nSummary Table:")
    print(f"{'Model':<12} {'AvgReturn(%)':>12} {'AvgDrawdown(%)':>15} {'AvgSharpe':>10} {'MaxReturn':>10} {'MinReturn':>10} {'N':>5}")
    for model, stats in summaries.items():
        print(f"{model:<12} {stats['avg_return']:12.2f} {stats['avg_drawdown']:15.2f} {stats['avg_sharpe']:10.2f} {stats['max_return']:10.2f} {stats['min_return']:10.2f} {stats['n']:5}")

def plot_boxplots(results, baseline_return):
    # Boxplot of returns without outlier markers
    data = [df['return_pct'] for df in results.values()]
    # Capitalize model names for labels
    labels = [label.capitalize() if label != 'finbert' else 'FinBERT' for label in results.keys()]
    plt.figure(figsize=(8,5))
    # Showfliers=False removes the random circles (outliers)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.axhline(baseline_return, color='r', linestyle='--', label='Baseline')
    plt.ylabel('Return (%)')
    plt.xlabel('Transformer Architecture')
    plt.title('Distribution of Returns by Model')
    plt.legend()
    plt.tight_layout()
    plt.savefig('returns_boxplot.png')
    plt.show()

def main():
    rl_results = load_rl_results()
    baseline_df = load_baseline()
    summaries = {}
    # RL agent summaries
    for model, df in rl_results.items():
        summaries[model] = summarize_rl(df)
    # Baseline summary
    if baseline_df is not None:
        summaries['finbert_baseline'] = summarize_baseline(baseline_df)
    print_summary_table(summaries)
    # Optionally plot
    try:
        if baseline_df is not None:
            plot_boxplots(rl_results, summaries['finbert_baseline']['avg_return'])
    except Exception as e:
        print(f"Plotting failed: {e}")
    # Save summary CSV
    pd.DataFrame(summaries).T.to_csv('summary_analysis.csv')
    print("\nSaved summary to summary_analysis.csv and boxplot to returns_boxplot.png")

if __name__ == "__main__":
    main() 