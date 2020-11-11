from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from features import compute_features

file_names = ["a" + str(i).zfill(4) for i in range(11, 200)]
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

# making the classifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)

# training the classifier
model = clf.fit(X_train, y_train)

# use a0010 as for testing
y_predict = model.predict(X_test)
print(y_predict)
print(y_test)
num_correct = sum([1 if y_test[i] == y_predict[i] else 0 for i in range(len(y_test))])
accuracy = num_correct / len(y_test)
print(accuracy) 


