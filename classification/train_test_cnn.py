from cnn_preprocess import CNNPreprocess
from cnn import CNN
import numpy as np

# Driver script to train and test a CNN model.

def test_cnn(file_sets):
    cnn_preprocess = CNNPreprocess(file_sets=file_sets)
    data = cnn_preprocess.preprocess_data()

    X = np.array(data["values"])
    y = np.array(data["labels"])

    cnn = CNN()
    test_error, test_accuracy = cnn.test_model(X, y)

    print("Accuracy on the test set is: {}".format(test_accuracy))
    print("Error on the test set is: {}".format(test_error))

def train_cnn(file_sets):
    cnn_preprocess = CNNPreprocess(file_sets=file_sets)
    data = cnn_preprocess.preprocess_data()

    X = np.array(data["values"])
    y = np.array(data["labels"])

    cnn = CNN()
    cnn.train_model(X, y)

file_sets = [
    (["./challenge_data/training-a/a" + str(i).zfill(4) for i in range(1, 410)], "./challenge_data/training-a/"),
    (["./challenge_data/training-b/b" + str(i).zfill(4) for i in range(1, 491)], "./challenge_data/training-b/"),
    (["./challenge_data/training-c/c" + str(i).zfill(4) for i in range(1, 32)], "./challenge_data/training-c/"),
    (["./challenge_data/training-d/d" + str(i).zfill(4) for i in range(1, 56)], "./challenge_data/training-d/"),
    (["./challenge_data/training-e/e" + str(i).zfill(5) for i in range(1, 2142)], "./challenge_data/training-e/"),
    (["./challenge_data/training-f/f" + str(i).zfill(4) for i in range(1, 115)], "./challenge_data/training-f/"),
]

train_cnn(file_sets)