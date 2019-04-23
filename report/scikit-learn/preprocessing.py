from sklearn import preprocessing
import numpy as np

data = np.array([[2.2, 5.9, -1.8], [5.4, -3.2, -5.1], [-1.9, 4.2, 3.2]])
bindata = preprocessing.Binarizer(threshold=1.5).transform(data)

#print('Binarized data:\n\n', bindata)

print('Mean (before)= ', data.mean(axis=0))
print('Standard Deviation (before)= ', data.std(axis=0))

scaled_data = preprocessing.scale(data)
print('Mean (after)= ', scaled_data.mean(axis=0))
print('Standard Deviation (after)= ', scaled_data.std(axis=0))

minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_minmax = minmax_scaler.fit_transform(data)
print('MinMaxScaler applied on the data:\n\n', data_minmax)

data_l1 = preprocessing.normalize(data, norm='l1')
data_l2 = preprocessing.normalize(data, norm='l2')

print('L1 normalized data:\n\n', data_l1)
print('L2 normalized data:\n\n', data_l2)