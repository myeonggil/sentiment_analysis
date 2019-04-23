from sklearn import preprocessing

labels = ['setosa', 'versicolor', 'virginica']
encoder = preprocessing.LabelEncoder()
encoder.fit(labels)

for i, items in enumerate(encoder.classes_):
    print(items, ' => ', i)

more_labels = ['versicolor', 'versicolor', 'virginica', 'setosa', 'versicolor']
more_labels_encoded = encoder.transform(more_labels)

print('More labels = ', more_labels)
print('More labels Encoding = ', list(more_labels_encoded))
