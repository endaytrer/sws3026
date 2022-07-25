import joblib
import numpy as np
from matplotlib import pyplot as plt

from Classifier import Classifier

clf: Classifier = Classifier()

nomask_data_path = "output/dataset_nomask"
x = []
y = []
for i in range(50):
    for j in range(15):
        filepath = f"{nomask_data_path}/s{i}/{j}.jpg"
        img = plt.imread(filepath)
        x.append(img)
        y.append(i)

x, y = np.array(x), np.array(y)
x = np.mean(x, axis=3)
dimensions, width, height = x.shape
x = x.reshape((dimensions, width * height))
clf.fit(x, y, num_features=40)
joblib.dump(clf, 'unmask_model.joblib')
