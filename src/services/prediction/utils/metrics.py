from time import time
from math import sqrt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 自定义MAPE函数以兼容旧版本sklearn
def mean_absolute_percentage_error(y_true, y_pred):
    """计算平均绝对百分比误差"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-10  # 避免除零
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


METRIC_DICT = {
    'mae': mean_absolute_error,
    'MAE': mean_absolute_error,
    
    'rmse': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
    'RMSE': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
    
    'r2': r2_score,
    'R2': r2_score,
    
    'mape': mean_absolute_percentage_error,
    'MAPE': mean_absolute_percentage_error,
    
    'mse': mean_squared_error,
    'MSE': mean_squared_error,
}
VALID_METRICS = list(METRIC_DICT.keys())


class AverageRecord(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class LossRecord:
    def __init__(self, loss_list):
        self.start_time = time()
        self.loss_list = loss_list
        self.loss_dict = {loss: AverageRecord() for loss in self.loss_list}
    
    def update(self, update_dict, n):
        for key, value in update_dict.items():
            self.loss_dict[key].update(value, n)
    
    def format_metrics(self):
        result = ""
        for loss in self.loss_list:
            result += "{}: {:.8f} | ".format(loss, self.loss_dict[loss].avg)
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result
    
    def to_dict(self):
        return {
            loss: self.loss_dict[loss].avg for loss in self.loss_list
        }
    
    def __str__(self):
        return self.format_metrics()
    
    def __repr__(self):
        return self.loss_dict[self.loss_list[0]].avg
