#!/usr/bin/env python3
"""
DiffSR Inference Script
Executes super-resolution inference using trained models and generates reports.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Add DiffSR root to path
diffsr_root = Path(__file__).parent
sys.path.insert(0, str(diffsr_root))

from forecastors.base import BaseForecaster
from forecastors.ddpm import DDPMForecaster
from forecastors.resshift import ResshiftForecaster
from report_generator import DiffSRReportGenerator


FORECASTER_DICT = {
    'base': BaseForecaster,
    'ddpm': DDPMForecaster,
    'resshift': ResshiftForecaster,
}


def main():
    parser = argparse.ArgumentParser(description='DiffSR Inference')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model directory (contains best_model.pth and config.yaml)')
    parser.add_argument('--forecastor_type', type=str, default='base',
                        choices=['base', 'ddpm', 'resshift'],
                        help='Forecastor type')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='Dataset split to evaluate')

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"DiffSR Inference")
    print(f"{'='*60}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Forecastor Type: {args.forecastor_type}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Evaluation Split: {args.split}")
    print(f"{'='*60}\n")

    # Check model directory exists
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Error: Model directory does not exist: {model_dir}")
        sys.exit(1)

    model_path = model_dir / 'best_model.pth'
    config_path = model_dir / 'config.yaml'

    if not model_path.exists():
        print(f"❌ Error: Model checkpoint not found: {model_path}")
        sys.exit(1)

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    print("✓ Model checkpoint found")
    print("✓ Config file found\n")

    # Initialize forecaster
    print(f"🔧 Initializing {args.forecastor_type} forecaster...")
    try:
        forecaster_class = FORECASTER_DICT[args.forecastor_type]
        forecaster = forecaster_class(str(model_dir))
        print("✓ Forecaster initialized\n")
    except Exception as e:
        print(f"❌ Error initializing forecaster: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Build data loaders
    print("📦 Building data loaders...")
    try:
        forecaster.data_name = forecaster.data_args['name']
        forecaster.build_data()
        print(f"✓ Data loaders built")
        print(f"  - Train samples: {len(forecaster.train_loader.dataset)}")
        print(f"  - Valid samples: {len(forecaster.valid_loader.dataset)}")
        print(f"  - Test samples: {len(forecaster.test_loader.dataset)}\n")
    except Exception as e:
        print(f"❌ Error building data loaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Select loader based on split
    if args.split == 'train':
        loader = forecaster.train_loader
    elif args.split == 'valid':
        loader = forecaster.valid_loader
    else:
        loader = forecaster.test_loader

    # Run inference
    print(f"🚀 Running inference on {args.split} set...")
    print(f"{'='*60}")
    try:
        loss_record = forecaster.forecast(loader, forecaster.normalizer)
        print(f"{'='*60}\n")
        print("✅ Inference completed successfully!\n")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Convert metrics to dict
    metrics_dict = loss_record.to_dict()

    # Print metrics
    print("📊 Inference Metrics:")
    print(f"{'='*60}")
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # Save metrics to JSON
    metrics_file = output_dir / 'metrics.json'

    # Prepare complete metrics for report
    complete_metrics = {
        'best_psnr': metrics_dict.get('psnr', 0.0),
        'best_ssim': metrics_dict.get('ssim', 0.0),
        'best_loss': metrics_dict.get('mse', 0.0),
        'test_metrics': {
            'psnr': metrics_dict.get('psnr', 0.0),
            'ssim': metrics_dict.get('ssim', 0.0),
            'mse': metrics_dict.get('mse', 0.0),
            'rmse': metrics_dict.get('rmse', 0.0),
            'rel_l2': 0.0,  # Not computed by default evaluator
        },
        'model_path': str(model_path),
        'forecastor_type': args.forecastor_type,
        'split': args.split,
    }

    with open(metrics_file, 'w') as f:
        json.dump(complete_metrics, f, indent=2)

    print(f"💾 Metrics saved to: {metrics_file}\n")

    # Generate report
    print("📝 Generating inference report...")
    try:
        # Load config from model directory
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Generate report
        generator = DiffSRReportGenerator()
        report_path = output_dir / 'inference_report.md'

        generator.generate_train_report(
            config=config,
            metrics=complete_metrics,
            output_path=str(report_path)
        )

        print(f"✅ Report generated: {str(report_path)}\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate report: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)
    print("✅ Inference pipeline completed successfully!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Report: {output_dir / 'inference_report.md'}")


if __name__ == '__main__':
    main()
