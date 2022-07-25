import numpy as np
import sklearn.svm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

weight_power = 4


class ModifiedClassifier:

    u: np.ndarray
    sigma: np.ndarray
    v: np.ndarray
    svm: sklearn.svm.SVC

    def __init__(self, unmask_x, num_features) -> None:
        self.svm = sklearn.svm.SVC()
        temp_u, temp_sigma, temp_v = np.linalg.svd(
            unmask_x.T, full_matrices=False)
        self.u = temp_u[:, :num_features]
        self.sigma = temp_sigma[:num_features]
        self.v = temp_v[:num_features, :]

    def decompose(self, mask_x) -> np.ndarray:
        # new decompose method
        test_v = []
        weights = (1 - mask_x.T ** weight_power)
        x_copy = mask_x.copy()
        # since the new eigenfaces are no longer orthogonal, we decompose mask_x progressively, subtracting original result each time.
        for i in range(self.u.shape[1]):
            prod = np.sum((self.u[:, i].reshape((weights.shape[0], 1)) *
                           weights) * x_copy.T, axis=0)
            x_copy -= prod.reshape((prod.shape[0], 1)
                                   ) @ self.u[:, i].reshape(1, self.u.shape[0])
            test_v.append(prod / self.sigma[i])
        test_v = np.array(test_v)

        # # original decompose model

        # test_v = (self.u.T @ ((1 - mask_x.T ** weight_power) * mask_x.T)) / \
        #     self.sigma.reshape((self.sigma.shape[0], 1))

        return test_v

    def fit(self, mask_x, y) -> None:
        test_v = self.decompose(mask_x)
        self.svm.fit(test_v.T, y)

    def predict(self, mask_x: np.ndarray) -> np.ndarray:
        test_v = self.decompose(mask_x)
        return self.svm.predict(test_v.T)


unmask_data_path = "output/dataset_nomask"
mask_data_path = "output/dataset_mask"
unmask_x = []
mask_x = []
y = []

for i in range(50):
    for j in range(15):
        unmask_img = plt.imread(f"{unmask_data_path}/s{i}/{j}.jpg")
        mask_img = plt.imread(f"{mask_data_path}/s{i}/{j}.jpg")
        unmask_x.append(unmask_img)
        mask_x.append(mask_img)
        y.append(i)

unmask_x, mask_x, y = np.array(
    unmask_x, np.float64), np.array(mask_x, np.float64), np.array(y)
unmask_x, mask_x = np.mean(unmask_x, axis=3) / \
    255, np.mean(mask_x, axis=3) / 255
dimensions, width, height = unmask_x.shape
unmask_x, mask_x = unmask_x.reshape(
    (dimensions, width * height)), mask_x.reshape((dimensions, width * height))
clf = ModifiedClassifier(unmask_x, 70)
x_train, x_test, y_train, y_test = train_test_split(
    mask_x, y, train_size=0.8, test_size=0.2)
clf.fit(unmask_x, y)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
