from classifier import Classifier

# Driver script to train and test the AdaBoost classifier.

def test_classifier(file_sets, hyperparameter_optimization = True):
    """
    Runs an AdaBoost classifier on audio files and applies k-fold validation. Use
    this method for testing the model.
    
    Args:
        file_sets (array<tuple(filepaths, prefix)>): path to files to train the classifier
            e.g. [
                (["./challenge_data/a1.wav", "./challenge_data/a2.wav", ...], "./challenge_data/"),
                ...
            ]
        hyperparameter_optimization (bool): boolean indicating whether to apply hyperparameter optimization or not (defaults to True)
    Returns:
        accuracy, std_dev (tuple): the accuracy and standard deviation of the metrics of the model
    """
    
    classifier = Classifier(file_sets)
    X, y = classifier.load_data()
    model = classifier.get_model(hyperparameter_optimization)
    accuracy, std_dev = classifier.evaluate_model(X, y, model)
    return accuracy, std_dev

def train_classifier(file_sets, hyperparameter_optimization = True, model_filename = "adaboost_classifier.sav"):
    """
    Trains an AdaBoost model. Use this method to train a model on a full dataset.
    
    Args:
        file_sets (array<tuple(filepaths, prefix)>): path to files to train the classifier
            e.g. [
                (["./challenge_data/a1.wav", "./challenge_data/a2.wav", ...], "./challenge_data/"),
                ...
            ]
        hyperparameter_optimization (bool): boolean indicating whether to apply hyperparameter optimization or not (defaults to True)
        model_filename (string): location to store the model
    """

    classifier = Classifier(file_sets)
    X, y = classifier.load_data()
    model = classifier.get_model(hyperparameter_optimization)
    classifier.train_model(model, X, y, model_filename)

file_sets = [
    (["./challenge_data/training-a/a" + str(i).zfill(4) for i in range(1, 410)], "./challenge_data/training-a/"),
    (["./challenge_data/training-b/b" + str(i).zfill(4) for i in range(1, 491)], "./challenge_data/training-b/"),
    (["./challenge_data/training-c/c" + str(i).zfill(4) for i in range(1, 32)], "./challenge_data/training-c/"),
    (["./challenge_data/training-d/d" + str(i).zfill(4) for i in range(1, 56)], "./challenge_data/training-d/"),
    (["./challenge_data/training-e/e" + str(i).zfill(5) for i in range(1, 2142)], "./challenge_data/training-e/"),
    (["./challenge_data/training-f/f" + str(i).zfill(4) for i in range(1, 115)], "./challenge_data/training-f/"),
]

accuracy, std_dev = test_classifier(file_sets)
print("Accuracy: ", accuracy)
print("Standard Deviation: ", std_dev)