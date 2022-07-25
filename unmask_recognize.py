import numpy as np
from matplotlib import pyplot as plt
from Classifier import Classifier
from Preprocessor import Preprocessor
import joblib


data_path = "output/dataset_nomask"

if __name__ == "__main__":
    preprocessor = Preprocessor('shape_predictor_68_face_landmarks.dat')
    model: Classifier = joblib.load("model.joblib")
    images = np.array([plt.imread('Dataset_1/s34/09.jpg'),
                       plt.imread('Dataset_1/s12/02.jpg')])
    processed = np.mean(images, axis=3).reshape((2, 22500))
    print(model.predict(processed))
