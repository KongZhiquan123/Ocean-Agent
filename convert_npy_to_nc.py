"""
Convert ERA5 wind .npy data to NetCDF format for preprocessing
"""
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta

# Load data
print("Loading ERA5 wind data...")
data = np.load('ERA5wind_vo_128_128_subset_10000.npy')
print(f"Data shape: {data.shape}")
print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")

# Remove last dimension if it's 1
if data.shape[-1] == 1:
    data = data.squeeze(-1)  # Shape: (10000, 128, 128)
print(f"Reshaped data: {data.shape}")

# Create NetCDF file
output_file = 'ERA5wind_vo_subset.nc'
print(f"\nCreating NetCDF file: {output_file}")

dataset = nc.Dataset(output_file, 'w', format='NETCDF4')

# Create dimensions
time_dim = dataset.createDimension('time', data.shape[0])
lat_dim = dataset.createDimension('lat', data.shape[1])
lon_dim = dataset.createDimension('lon', data.shape[2])

# Create coordinate variables
times = dataset.createVariable('time', 'f8', ('time',))
lats = dataset.createVariable('lat', 'f4', ('lat',))
lons = dataset.createVariable('lon', 'f4', ('lon',))

# Create data variable
vo = dataset.createVariable('vo', 'f4', ('time', 'lat', 'lon'), 
                            zlib=True, complevel=4)

# Set attributes
times.units = 'hours since 2000-01-01 00:00:00'
times.calendar = 'gregorian'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
vo.units = 's**-1'
vo.long_name = 'Vorticity (relative)'
vo.standard_name = 'atmosphere_relative_vorticity'

# Fill coordinate data
# Assume global coverage for 128x128
lats[:] = np.linspace(-90, 90, 128)
lons[:] = np.linspace(-180, 180, 128)
times[:] = np.arange(data.shape[0])  # hourly data

# Fill data
print("Writing data to NetCDF...")
vo[:, :, :] = data

# Add global attributes
dataset.description = 'ERA5 wind vorticity data converted from NPY'
dataset.source = 'ERA5wind_vo_128_128_subset_10000.npy'
dataset.history = f'Created {datetime.now().isoformat()}'
dataset.Conventions = 'CF-1.6'

# Close file
dataset.close()
print(f"[SUCCESS] NetCDF file created: {output_file}")
print(f"   Variables: {list(nc.Dataset(output_file, 'r').variables.keys())}")
print(f"   Shape: time={data.shape[0]}, lat={data.shape[1]}, lon={data.shape[2]}")
