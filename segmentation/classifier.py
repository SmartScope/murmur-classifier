from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from features import compute_features

# get features
X = []
for fname in ("a0001", "a0002", "a0003", "a0004", "a0005", "a0006", "a0007", "a0008", "a0009"):
    features = compute_features(fname)['arr']   
    X.append(features)

# get labels
abnormal_records = set()
with open("RECORDS-abnormal") as fp:
   for line in fp:
       l = line.rstrip("\n")
       abnormal_records.add(l)

# 1 means abnormal, 0 means normal
y = [1 if fname in abnormal_records else 0 for fname in ("a0001", "a0002", "a0003", "a0004", "a0005", "a0006", "a0007", "a0008", "a0009")]

# for each sample, need array of features

# making the classifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)

# training the classifier
model = clf.fit(X, y)

# use a0010 as for testing
a10 = compute_features("a0010")['arr']   
print(model.predict([a10]))

