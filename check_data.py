import numpy as np

data = np.load('D:/tmp/ERA5wind_vo_128_128_subset_10000.npy')
print('数据形状:', data.shape)
print('数据类型:', data.dtype)
print('数值范围: [{:.4f}, {:.4f}]'.format(data.min(), data.max()))
print('平均值: {:.4f}'.format(data.mean()))
