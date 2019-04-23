import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

timesteps = seq_length = 7
data_dim = 5
output_dim = 1

xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]
xy = MinMaxScaler(xy, feature_range=(0, 1))
x = xy
y = xy[:, [-1]]

print(x)
print(y)

dataX = []
dataY = []