#!/usr/bin/env python3
"""
JAXA SST Data Preprocessing Script
预处理 JAXA 卫星 SST 数据用于超分辨率模型训练
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

def inspect_nc_file(filepath):
    """检查 NetCDF 文件结构"""
    try:
        import netCDF4
        ds = netCDF4.Dataset(filepath, 'r')
        info = {
            'dimensions': dict(ds.dimensions),
            'variables': list(ds.variables.keys()),
            'attrs': dict(ds.attrs),
        }
        
        for var in ds.variables.keys():
            var_obj = ds.variables[var]
            info[f'{var}_shape'] = var_obj.shape
            info[f'{var}_dtype'] = str(var_obj.dtype)
            
        ds.close()
        return info
    except Exception as e:
        # Try xarray as fallback
        try:
            import xarray as xr
            ds = xr.open_dataset(filepath)
            info = {
                'dims': dict(ds.dims),
                'vars': list(ds.data_vars),
                'coords': list(ds.coords),
            }
            ds.close()
            return info
        except Exception as e2:
            return {'error': str(e2)}

def preprocess_sst_data(input_dir, output_dir, variable_name='sst'):
    """
    预处理 JAXA SST 数据
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .nc files
    input_path = Path(input_dir)
    nc_files = sorted(input_path.glob('*.nc'))
    
    if not nc_files:
        print(f"ERROR: No .nc files found in {input_dir}")
        return False
    
    print(f"Found {len(nc_files)} NetCDF files")
    
    # Inspect first file
    print("\n=== 数据文件检查 ===")
    for nc_file in nc_files[:3]:  # 检查前 3 个文件
        print(f"\n文件: {nc_file.name}")
        info = inspect_nc_file(str(nc_file))
        if 'error' not in info:
            print(f"  维度: {info.get('dimensions', info.get('dims', 'N/A'))}")
            print(f"  变量: {info.get('variables', info.get('vars', 'N/A'))}")
        else:
            print(f"  错误: {info['error']}")
    
    print("\n=== 预处理配置 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标变量: {variable_name}")
    print(f"输入文件数: {len(nc_files)}")
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'variable': variable_name,
        'input_files': [f.name for f in nc_files],
        'file_count': len(nc_files),
        'status': 'initial_inspection',
    }
    
    # Save inspection results
    report_path = os.path.join(output_dir, 'preprocessing_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ 预处理报告已保存: {report_path}")
    
    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python preprocess_jaxa_data.py <input_dir> <output_dir> [variable_name]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    variable_name = sys.argv[3] if len(sys.argv) > 3 else 'sst'
    
    success = preprocess_sst_data(input_dir, output_dir, variable_name)
    sys.exit(0 if success else 1)

