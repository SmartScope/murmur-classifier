from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from features import compute_features
import pickle
import time
import matplotlib.pyplot as plt

def get_training_testing_data():
    file_names = ["a" + str(i).zfill(4) for i in range(11, 61)]
    # get features
    X = []
    for fname in file_names:
        features = compute_features(fname)['arr']   
        X.append(features)
        
    # get labels
    abnormal_records = set()
    with open("RECORDS-abnormal") as fp:
        for line in fp:
            l = line.rstrip("\n")
            abnormal_records.add(l)

    # 1 means abnormal, 0 means normal
    y = [1 if fname in abnormal_records else 0 for fname in file_names]

    # Split train & test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
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
    filename = f'{int(time.time())}.sav'
    pickle.dump(model, open(filename, 'wb'))

def invoke_model(filename, X_train, X_test, y_train, y_test):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)

t = time.time()
X_train, X_test, y_train, y_test = get_training_testing_data()
f = time.time()
print("feature extraction took ", f-t)
train_model(X_train, X_test, y_train, y_test)
print("training took ", time.time() - f)
# invoke_model("1605112488.sav", X_train, X_test, y_train, y_test)


