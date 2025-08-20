#!/usr/bin/env python3
"""
Model comparison pipeline for Walmart Sales Forecasting.

This pipeline:
1. Loads backtest results from all models
2. Aggregates metrics across models
3. Generates comparison reports and visualizations
4. Saves aggregated results
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.logging_utils import setup_logging, get_logger
from src.paths import METRICS_DIR, FIGURES_DIR
from src.utils.io_utils import safe_read_csv

def main():
    """Main model comparison pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare all models for Walmart sales forecasting')
    parser.add_argument('--metrics-dir', type=str, default=None, help='Directory containing metrics')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('compare_all', 'INFO')
    
    try:
        logger.info("Starting model comparison pipeline")
        
        # Step 1: Load metrics data
        logger.info("Step 1: Loading metrics data")
        metrics_dir = Path(args.metrics_dir) if args.metrics_dir else METRICS_DIR
        
        # Load model comparison data
        comparison_path = metrics_dir / "model_comparison.csv"
        if not comparison_path.exists():
            logger.error(f"Model comparison file not found at {comparison_path}")
            logger.info("Please run the backtest_all pipeline first")
            return
        
        comparison_df = pd.read_csv(comparison_path)
        logger.info(f"Loaded comparison data: {comparison_df.shape}")
        
        # Step 2: Load backtest summary
        logger.info("Step 2: Loading backtest summary")
        summary_path = metrics_dir / "backtest_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary_stats = json.load(f)
            logger.info(f"Loaded summary stats for {len(summary_stats)} models")
        else:
            summary_stats = {}
            logger.warning("Backtest summary not found")
        
        # Step 3: Generate aggregated metrics
        logger.info("Step 3: Generating aggregated metrics")
        
        # Overall model performance
        overall_metrics = comparison_df.groupby('model').agg({
            'mae': ['mean', 'std', 'min', 'max'],
            'rmse': ['mean', 'std', 'min', 'max'],
            'mape': ['mean', 'std', 'min', 'max'],
            'smape': ['mean', 'std', 'min', 'max'],
            'wape': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Flatten column names
        overall_metrics.columns = ['_'.join(col).strip() for col in overall_metrics.columns]
        overall_metrics = overall_metrics.reset_index()
        
        logger.info("Overall metrics calculated successfully")
        
        # Step 4: Generate fold-level analysis
        logger.info("Step 4: Generating fold-level analysis")
        
        # Calculate fold stability (coefficient of variation)
        fold_stability = comparison_df.groupby('model').agg({
            'mae': lambda x: x.std() / x.mean() * 100,
            'rmse': lambda x: x.std() / x.mean() * 100,
            'mape': lambda x: x.std() / x.mean() * 100
        }).round(2)
        
        fold_stability.columns = ['mae_cv', 'rmse_cv', 'mape_cv']
        fold_stability = fold_stability.reset_index()
        
        logger.info("Fold stability analysis completed")
        
        # Step 5: Generate ranking analysis
        logger.info("Step 5: Generating ranking analysis")
        
        # Rank models by each metric
        ranking_analysis = {}
        metrics_to_rank = ['mae', 'rmse', 'mape', 'smape', 'wape']
        
        for metric in metrics_to_rank:
            if metric in comparison_df.columns:
                # Calculate mean rank across folds
                metric_ranks = comparison_df.groupby('fold')[metric].rank()
                avg_ranks = metric_ranks.groupby(comparison_df['model']).mean().sort_values()
                
                ranking_analysis[metric] = avg_ranks.to_dict()
        
        logger.info("Ranking analysis completed")
        
        # Step 6: Save comparison results
        logger.info("Step 6: Saving comparison results")
        
        output_dir = Path(args.output_dir) if args.output_dir else METRICS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save overall metrics
        overall_path = output_dir / "overall_model_metrics.csv"
        overall_metrics.to_csv(overall_path, index=False)
        logger.info(f"Overall metrics saved to {overall_path}")
        
        # Save fold stability
        stability_path = output_dir / "fold_stability_analysis.csv"
        fold_stability.to_csv(stability_path, index=False)
        logger.info(f"Fold stability analysis saved to {stability_path}")
        
        # Save ranking analysis
        ranking_path = output_dir / "model_rankings.json"
        with open(ranking_path, 'w') as f:
            json.dump(ranking_analysis, f, indent=2)
        logger.info(f"Model rankings saved to {ranking_path}")
        
        # Step 7: Generate summary report
        logger.info("Step 7: Generating summary report")
        
        summary_report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': comparison_df['model'].nunique(),
            'total_folds': comparison_df['fold'].nunique(),
            'metrics_evaluated': metrics_to_rank,
            'best_model_by_metric': {},
            'overall_winner': None
        }
        
        # Find best model by each metric
        for metric in metrics_to_rank:
            if metric in comparison_df.columns:
                best_model = comparison_df.groupby('model')[metric].mean().idxmin()
                best_score = comparison_df.groupby('model')[metric].mean().min()
                summary_report['best_model_by_metric'][metric] = {
                    'model': best_model,
                    'score': round(best_score, 2)
                }
        
        # Determine overall winner (average rank across all metrics)
        if ranking_analysis:
            overall_ranks = {}
            for model in comparison_df['model'].unique():
                model_ranks = []
                for metric, ranks in ranking_analysis.items():
                    if model in ranks:
                        model_ranks.append(ranks[model])
                
                if model_ranks:
                    overall_ranks[model] = np.mean(model_ranks)
            
            if overall_ranks:
                overall_winner = min(overall_ranks, key=overall_ranks.get)
                summary_report['overall_winner'] = overall_winner
        
        # Save summary report
        report_path = output_dir / "comparison_summary.json"
        with open(report_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        logger.info(f"Summary report saved to {report_path}")
        
        # Step 8: Print comparison summary
        logger.info("Model Comparison Summary:")
        logger.info(f"  Models evaluated: {summary_report['models_evaluated']}")
        logger.info(f"  Total folds: {summary_report['total_folds']}")
        
        for metric, best in summary_report['best_model_by_metric'].items():
            logger.info(f"  Best {metric.upper()}: {best['model']} ({best['score']})")
        
        if summary_report['overall_winner']:
            logger.info(f"  Overall winner: {summary_report['overall_winner']}")
        
        logger.info("Model comparison pipeline completed successfully")
        
        return {
            'overall_metrics': overall_metrics,
            'fold_stability': fold_stability,
            'ranking_analysis': ranking_analysis,
            'summary_report': summary_report
        }
        
    except Exception as e:
        logger.error(f"Model comparison pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
