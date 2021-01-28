from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import plot_confusion_matrix
from features import FeaturesProcessor
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

class Classifier:
    def __init__(self, folders):
        self.folders = folders

    def remove_prefix(self, text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text

    def load_data(self):
        file_names = ["./challenge_data/b" + str(i).zfill(4) for i in range(1, 491)]

        # get features
        X = []
        for fname in file_names:
            features_processor = FeaturesProcessor(fname)
            features = features_processor.get_all_features()
            X.append(features)
        
        # get labels
        abnormal_records = set()
        with open("./challenge_data/RECORDS-abnormal") as fp:
            for line in fp:
                l = line.rstrip("\n")
                abnormal_records.add(l)

        # 1 means abnormal, 0 means normal
        y = [1 if self.remove_prefix(fname, "./challenge_data/") in abnormal_records else 0 for fname in file_names]

        return X, y

    def get_model(self, hyperparamter_optimization=False):
        if hyperparamter_optimization:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.001, 0.01, 0.2, 0.5]
            }
            
            model = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid)
            return model

        return AdaBoostClassifier()

    def evaluate_model(self, X, y, model):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        return np.mean(n_scores), np.std(n_scores)

    def train_model(self, clf, X, y):
        # Train the classifier
        model = clf.fit(X, y)

        # save the model to disk
        filename = f'{int(time.time())}.sav'
        pickle.dump(model, open(filename, 'wb'))

    def invoke_model(self, filename, X):
        # Load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(X)
        print(result)

    def predict(self, filename):
        return 0

classifier = Classifier(folders="")
X, y = classifier.load_data()
model = classifier.get_model(hyperparamter_optimization=True)
accuracy, std_dev = classifier.evaluate_model(X, y, model)
print("Accuracy: ", accuracy)
print("Standard Deviation: ", std_dev)


