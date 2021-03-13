from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from classification.features import FeaturesProcessor
import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from classification.segmentation_util import Base

class Classifier(Base):
    def __init__(self, file_sets = None):
        self.file_sets = file_sets

    def load_data(self):
        X_overall = []
        y_overall = []

        for pair in self.file_sets:
            file_set, prefix = pair[0], pair[1]

            # Get features
            X = []
            for fname in file_set:
                print(fname)
                features_processor = FeaturesProcessor(fname)
                features = features_processor.get_all_features()
                X.append(features)
            
            X_overall += X

            # Get labels
            abnormal_records = set()

            with open("{prefix}RECORDS-abnormal".format(prefix=prefix)) as fp:
                for line in fp:
                    l = line.rstrip("\n")
                    abnormal_records.add(l)
            
            # 1 means abnormal, 0 means normal
            y = [1 if self.remove_prefix(fname, prefix) in abnormal_records else 0 for fname in file_set]
            y_overall += y
        
        return X_overall, y_overall

    def get_model(self, hyperparamter_optimization = True):
        if not hyperparamter_optimization:
            return AdaBoostClassifier()
        
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.001, 0.01, 0.2, 0.5]
        }
        
        model = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid)
        return model

    def classification_report_with_accuracy_score(self, y_true, y_pred):
        print(classification_report(y_true, y_pred)) # print classification report
        return accuracy_score(y_true, y_pred) # return accuracy score

    def evaluate_model(self, X, y, model):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
        n_scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(self.classification_report_with_accuracy_score))
        return np.mean(n_scores), np.std(n_scores)

    def train_model(self, clf, X, y, filename = "adaboost_classifier.sav"):
        model = clf.fit(X, y)
        pickle.dump(model, open(filename, 'wb'))

    def predict(self, X, ensemble = False, model_path = "./adaboost_classifier.sav"):
        loaded_model = pickle.load(open(model_path, 'rb'))
        if ensemble:
            return loaded_model.predict_proba(X)
        
        return loaded_model.predict(X)
