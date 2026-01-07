#!/usr/bin/env python3
"""VLM Benchmark Results Analyzer - Interactive Visualization"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
from pathlib import Path
import argparse


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load benchmark data and filter out failed runs."""
    df = pd.read_csv(csv_path)

    # Filter out failed runs
    df = df[df['success_rate'] > 0]
    df = df[df['avg_tokens_per_second'] > 0]

    # Extract short model name
    df['model_name'] = df['model_id'].apply(lambda x: x.split('/')[-1])

    # Aggregate duplicate runs (same model+platform)
    agg_cols = {
        'avg_tokens_per_second': 'mean',
        'avg_ttft_ms': 'mean',
        'avg_total_latency_ms': 'mean',
        'success_rate': 'mean',
        'peak_vram_mb': 'max',
        'total_tokens_generated': 'sum',
    }

    df_agg = df.groupby(['platform', 'model_name']).agg(agg_cols).reset_index()
    df_agg = df_agg.sort_values('avg_tokens_per_second', ascending=False)

    return df_agg


def create_dashboard(df: pd.DataFrame, output_path: str):
    """Create interactive HTML dashboard with Plotly."""

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Throughput: Tokens per Second (Higher = Better)',
            'Time to First Token (Lower = Better)',
            'Total Latency (Lower = Better)',
            'VRAM Usage (MB)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = {'vllm': '#636EFA', 'tgi': '#EF553B', 'tensorrt': '#00CC96'}

    # 1. Tokens per second (main metric)
    df_sorted = df.sort_values('avg_tokens_per_second', ascending=True)
    for platform in df['platform'].unique():
        mask = df_sorted['platform'] == platform
        fig.add_trace(
            go.Bar(
                y=df_sorted[mask]['model_name'],
                x=df_sorted[mask]['avg_tokens_per_second'],
                name=platform.upper(),
                orientation='h',
                marker_color=colors.get(platform, '#888'),
                text=df_sorted[mask]['avg_tokens_per_second'].round(1),
                textposition='outside',
                legendgroup=platform,
            ),
            row=1, col=1
        )

    # 2. Time to First Token
    df_ttft = df.sort_values('avg_ttft_ms', ascending=False)
    for platform in df['platform'].unique():
        mask = df_ttft['platform'] == platform
        fig.add_trace(
            go.Bar(
                y=df_ttft[mask]['model_name'],
                x=df_ttft[mask]['avg_ttft_ms'],
                name=platform.upper(),
                orientation='h',
                marker_color=colors.get(platform, '#888'),
                text=df_ttft[mask]['avg_ttft_ms'].round(0).astype(int).astype(str) + 'ms',
                textposition='outside',
                legendgroup=platform,
                showlegend=False,
            ),
            row=1, col=2
        )

    # 3. Total Latency
    df_latency = df.sort_values('avg_total_latency_ms', ascending=False)
    for platform in df['platform'].unique():
        mask = df_latency['platform'] == platform
        fig.add_trace(
            go.Bar(
                y=df_latency[mask]['model_name'],
                x=df_latency[mask]['avg_total_latency_ms'],
                name=platform.upper(),
                orientation='h',
                marker_color=colors.get(platform, '#888'),
                text=df_latency[mask]['avg_total_latency_ms'].round(0).astype(int).astype(str) + 'ms',
                textposition='outside',
                legendgroup=platform,
                showlegend=False,
            ),
            row=2, col=1
        )

    # 4. VRAM Usage
    df_vram = df.sort_values('peak_vram_mb', ascending=True)
    for platform in df['platform'].unique():
        mask = df_vram['platform'] == platform
        fig.add_trace(
            go.Bar(
                y=df_vram[mask]['model_name'],
                x=df_vram[mask]['peak_vram_mb'],
                name=platform.upper(),
                orientation='h',
                marker_color=colors.get(platform, '#888'),
                text=df_vram[mask]['peak_vram_mb'].round(0).astype(int).astype(str) + ' MB',
                textposition='outside',
                legendgroup=platform,
                showlegend=False,
            ),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        title={
            'text': 'VLM Benchmark Results - Speed Comparison',
            'font': {'size': 24}
        },
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        barmode='group'
    )

    # Update axes
    fig.update_xaxes(title_text="Tokens/sec", row=1, col=1)
    fig.update_xaxes(title_text="Milliseconds", row=1, col=2)
    fig.update_xaxes(title_text="Milliseconds", row=2, col=1)
    fig.update_xaxes(title_text="MB", row=2, col=2)

    # Save and open
    fig.write_html(output_path)
    return fig


def print_rankings(df: pd.DataFrame):
    """Print console summary with rankings."""
    print("\n" + "=" * 70)
    print("VLM BENCHMARK ANALYSIS - SPEED RANKINGS")
    print("=" * 70)

    # Top by tokens/sec
    print("\n### TOP 5 BY THROUGHPUT (Tokens/sec) ###")
    print("-" * 50)
    top5 = df.nlargest(5, 'avg_tokens_per_second')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"{i}. {row['model_name']:40} [{row['platform'].upper():5}]  {row['avg_tokens_per_second']:>7.1f} tok/s")

    # Top by TTFT
    print("\n### TOP 5 BY RESPONSIVENESS (Time to First Token) ###")
    print("-" * 50)
    top5_ttft = df.nsmallest(5, 'avg_ttft_ms')
    for i, (_, row) in enumerate(top5_ttft.iterrows(), 1):
        print(f"{i}. {row['model_name']:40} [{row['platform'].upper():5}]  {row['avg_ttft_ms']:>7.0f} ms")

    # Best overall (balance of speed and responsiveness)
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    best = df.loc[df['avg_tokens_per_second'].idxmax()]
    print(f"\nBest for THROUGHPUT:")
    print(f"  Model:    {best['model_name']}")
    print(f"  Platform: {best['platform'].upper()}")
    print(f"  Speed:    {best['avg_tokens_per_second']:.1f} tokens/sec")
    print(f"  TTFT:     {best['avg_ttft_ms']:.0f} ms")
    print(f"  Latency:  {best['avg_total_latency_ms']:.0f} ms")

    best_ttft = df.loc[df['avg_ttft_ms'].idxmin()]
    if best_ttft['model_name'] != best['model_name']:
        print(f"\nBest for RESPONSIVENESS:")
        print(f"  Model:    {best_ttft['model_name']}")
        print(f"  Platform: {best_ttft['platform'].upper()}")
        print(f"  TTFT:     {best_ttft['avg_ttft_ms']:.0f} ms")
        print(f"  Speed:    {best_ttft['avg_tokens_per_second']:.1f} tokens/sec")

    print("\n" + "=" * 70)

    # Full table
    print("\n### COMPLETE RANKINGS TABLE ###\n")
    print(f"{'Rank':<5} {'Model':<45} {'Platform':<8} {'Tok/s':>10} {'TTFT':>10} {'Latency':>12}")
    print("-" * 95)
    for i, (_, row) in enumerate(df.sort_values('avg_tokens_per_second', ascending=False).iterrows(), 1):
        print(f"{i:<5} {row['model_name']:<45} {row['platform'].upper():<8} {row['avg_tokens_per_second']:>10.1f} {row['avg_ttft_ms']:>8.0f}ms {row['avg_total_latency_ms']:>10.0f}ms")


def main():
    parser = argparse.ArgumentParser(description='Analyze VLM benchmark results')
    parser.add_argument('--input', '-i',
                        default='/home/shen/Downloads/results/benchmark_summary.csv',
                        help='Path to benchmark_summary.csv')
    parser.add_argument('--output', '-o',
                        default='/home/shen/Downloads/results/benchmark_analysis.html',
                        help='Output HTML file path')
    parser.add_argument('--no-open', action='store_true',
                        help='Do not auto-open browser')
    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    df = load_and_clean_data(args.input)

    print(f"Found {len(df)} valid benchmark results")

    print(f"\nGenerating visualization...")
    create_dashboard(df, args.output)
    print(f"Saved to: {args.output}")

    print_rankings(df)

    if not args.no_open:
        print(f"\nOpening in browser...")
        webbrowser.open(f'file://{args.output}')


if __name__ == '__main__':
    main()
