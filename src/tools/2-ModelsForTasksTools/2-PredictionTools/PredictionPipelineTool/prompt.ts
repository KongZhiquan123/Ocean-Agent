const DESCRIPTION = `🚀 Complete Prediction pipeline for ocean forecasting tasks.

**⚡ EMBEDDED FRAMEWORK - Use This Instead of Writing Code!**

This tool provides a complete, production-ready ocean prediction framework embedded in Kode.
When users ask for ocean forecasting or time series prediction, USE THIS TOOL!

## What's Included

✅ 10+ Pre-built Model Architectures:
   - OceanCNN - Fast ConvLSTM baseline
   - OceanResNet - ResNet backbone
   - OceanTransformer - Attention-based model
   - Fuxi (+ Light/Full/Auto variants) - Swin-Transformer based
   - NNG (+ variants) - Graph Neural Network
   - OneForecast (+ variants) - Lightweight GNN
   - GraphCast (+ variants) - DeepMind's approach
   - Fengwu (+ variants) - 2D+3D dual-path
   - Pangu (+ variants) - Huawei's weather model

✅ Multiple Dataset Types:
   - Ocean velocity data (H5, MAT, NetCDF formats)
   - Surface/mid-layer ocean data
   - Pearl River estuary data
   - Custom data with preprocessing

✅ Complete Training & Testing Pipelines:
   - YAML-based configuration
   - Multi-GPU training (DP/DDP)
   - Checkpoint management
   - Automatic logging and metrics
   - Visualization tools

## Operations

### 1. list_models
Show all available model architectures with descriptions.
**Use this first** to help users choose the right model.

### 2. list_configs
List available template configuration files.
Shows YAML configs in configs/ directory.

### 3. train
Train a prediction model with YAML config.
**This is the main training operation** - handles everything automatically.

Required: config_path (path to YAML config)
Optional: output_dir, epochs, batch_size, learning_rate, gpu_id

### 4. test
Run prediction on test data using trained model.
Loads trained model and generates forecasts.

Required: config_path, model_path
Optional: output_dir, gpu_id

## Key Features

🔧 **No External Dependencies**: Prediction code is embedded in Kode
📦 **Pre-configured Templates**: Ready-to-use YAML configs
🚀 **Production Ready**: Tested training and inference pipelines
🎯 **Multiple Domains**: Surface, mid-layer, estuary data
💪 **GPU Accelerated**: Automatic CUDA detection and usage
📊 **Comprehensive Metrics**: MSE, MAE, RMSE, R², MAPE

## Typical Workflows

### Training a New Model:
1. list_configs → Find appropriate template
2. Prepare data (H5/MAT/NetCDF format)
3. train → Run training with config
4. Monitor training progress in output_dir

### Running Prediction:
1. test → Generate forecasts using trained model
2. Visualize results with built-in tools

## Example - Train Fuxi Model:
{
  "operation": "train",
  "config_path": "configs/surface_config.yaml",
  "model_type": "Fuxi",
  "output_dir": "outputs/ocean_fuxi",
  "epochs": 100,
  "batch_size": 16
}

## Example - Run Prediction:
{
  "operation": "test",
  "config_path": "configs/surface_config.yaml",
  "model_path": "outputs/ocean_fuxi/best_model.pth",
  "output_dir": "results/ocean_forecast"
}

## When to Use This Tool

✅ User asks for "ocean prediction" or "forecasting"
✅ User wants to "train a model" for ocean/climate data
✅ User needs "time series prediction" for ocean data
✅ User mentions specific models (Fuxi, GraphCast, etc.)

❌ DON'T write custom training code from scratch
❌ DON'T create model definitions manually
✅ DO use this tool - it's complete and ready!

## Technical Details

- Training runs main.py from embedded Prediction directory
- Supports both single-GPU and distributed training
- Checkpoints saved automatically during training
- Test mode loads models and generates forecasts
- All paths relative to embedded Prediction location

## Note to AI Assistants

This tool wraps the complete Prediction framework. When users ask for ocean forecasting:
1. Suggest using this tool FIRST
2. Help them choose the right model with list_models
3. Guide them to appropriate config with list_configs
4. Run training/testing with this tool
5. Don't write custom Python training scripts!
6. When creating a new config file based on a template, ONLY modify the dataset path. Keep other parameters unchanged unless the user specifies otherwise.

The embedded code is production-tested and feature-complete.`

export default DESCRIPTION;