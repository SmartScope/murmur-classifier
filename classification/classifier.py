from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import plot_confusion_matrix
from features import compute_features
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_data():
    file_names = ["./challenge_data/c" + str(i).zfill(4) for i in range(1, 32)]

    # get features
    X = []
    for fname in file_names:
        features = compute_features(fname)['arr']   
        X.append(features)
        
    # get labels
    abnormal_records = set()
    with open("./challenge_data/RECORDS-abnormal") as fp:
        for line in fp:
            l = line.rstrip("\n")
            abnormal_records.add(l)

    # 1 means abnormal, 0 means normal
    y = [1 if remove_prefix(fname, "./challenge_data/") in abnormal_records else 0 for fname in file_names]

    return X, y

def get_training_testing_split(X, y, k=2):
    if k != 2:
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

        x_trains, x_tests, y_trains, y_tests = [], [], [], []
        for train_ix, test_ix in kfold.split(X, y):
            X_train, y_train, X_test, y_test = [], [], [], []
            for idx in train_ix:
                X_train.append(X[idx])
                y_train.append(y[idx])
            
            for idx in test_ix:
                X_test.append(X[idx])
                y_test.append(y[idx])
            
            x_trains.append(X_train)
            x_tests.append(X_test)
            y_trains.append(y_train)
            y_tests.append(y_test)
        
        return x_trains, x_tests, y_trains, y_tests

    # Default to train-test split when k = 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def model_validation(X, y, model=None):
    if model is None:
        model = AdaBoostClassifier()
    
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return np.mean(n_scores), np.std(n_scores)

def generate_hyperparameter_optimized_classifier():
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.001, 0.01, 0.2, 0.5]
    }

    clf = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid)
    return clf

def train_model(X_train, X_test, y_train, y_test, clf=None):
    if clf is None:
        # making the classifier
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)

    # training the classifier
    model = clf.fit(X_train, y_train)

    # use a0010 as for testing
    y_predict = model.predict(X_test)
    print("y_train: ", y_train)
    print("y_test",y_test)
    print("y_predict", y_predict)
    num_correct = sum([1 if y_test[i] == y_predict[i] else 0 for i in range(len(y_test))])
    accuracy = num_correct / len(y_test)
    print(accuracy)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()
    
    # save the model to disk
    # filename = f'{int(time.time())}.sav'
    # pickle.dump(model, open(filename, 'wb'))

def invoke_model(filename, X_train, X_test, y_train, y_test):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)

def test_kfold_work():
    x_trains, x_tests, y_trains, y_tests = get_training_testing_split(X, y, k=5)
    clf = generate_hyperparameter_optimized_classifier()
    for i in range(len(x_trains)):
        X_train, X_test, y_train, y_test = x_trains[i], x_tests[i], y_trains[i], y_tests[i]
        train_model(X_train, X_test, y_train, y_test, clf)

t = time.time()
X, y = get_data()
X_train, X_test, y_train, y_test = get_training_testing_split(X, y)

f = time.time()
print("feature extraction took ", f-t)
train_model(X_train, X_test, y_train, y_test)
print("training took ", time.time() - f)

# invoke_model("1605112488.sav", X_train, X_test, y_train, y_test)


