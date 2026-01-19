
import sys
import os
from pathlib import Path
import json
import numpy as np
import torch

# Add DiffSR to path
diffsr_path = Path("/home/marlon/kode-agent/kode/src/services/diffsr")
sys.path.insert(0, str(diffsr_path))

try:
    # Load input data
    input_path = "/home/marlon/OceanAgent/workspaces/bd1/bd22/datasets/era5_wind_prepared/test_lr.npy"
    if input_path.endswith('.npy'):
        data = np.load(input_path)
    elif input_path.endswith('.npz'):
        data_dict = np.load(input_path)
        data = data_dict[list(data_dict.keys())[0]]
    else:
        raise ValueError("Unsupported input format")

    # Load model checkpoint
    model_path = "/home/marlon/OceanAgent/workspaces/bd1/bd22/logs/fno_era5wind/ERA5wind/12_25/FNO2d_12_16_10/best_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')

    device = torch.device(f'cuda:undefined' if undefined >= 0 and torch.cuda.is_available() else 'cpu')

    result = {
        'status': 'inference_ready',
        'model': model_path,
        'input_shape': list(data.shape),
        'device': str(device),
        'checkpoint_keys': list(checkpoint.keys())[:5],
        'message': 'Model and data loaded. Use forecastors module for actual inference.'
    }

    
    output_dir = Path("/home/marlon/OceanAgent/workspaces/bd1/bd22/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    result['output_dir'] = str(output_dir)
    

    print(json.dumps(result, indent=2))

except Exception as e:
    print(json.dumps({'error': str(e)}), file=sys.stderr)
    sys.exit(1)
