from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from DataSplitter.splitindep import SplitData
from ModelBuilding.models import Model


class TuningModel:

    def __init__(self):
        self.data = SplitData()
        self.model = Model()

    def etctuner(self):
        x_train ,x_test, y_train, y_test = self.data.splitthedata()
        etc = self.model.etc()
        params = {
            'criterion' : ['gini', 'entropy'],
            'max_depth' : [i for i in range(1,10,1)],
            'max_features': ['sqrt', 'auto', 'log2', None],
            'n_estimators': [i for i in range(100, 3000, 30)],
            'min_samples_split': [i for i in range(2, 50, 1)],
            'min_samples_leaf': [i for i in range(1, 50, 1)]
        }
        randomcv_etc = RandomizedSearchCV(etc, param_distributions=params, n_jobs=-1, cv=5, n_iter=100, verbose=True)
        randomcv_etc.fit(x_train, y_train)
        etc_best = randomcv_etc.best_params_
        etc_clf = ExtraTreesClassifier(n_estimators=etc_best['n_estimators'],
                                       min_samples_leaf=etc_best['min_samples_leaf'],
                                       min_samples_split=etc_best['min_samples_split'],
                                       max_features=etc_best['max_features'],
                                       criterion=etc_best['criterion'], max_depth=etc_best['max_depth'])
        etc_clf.fit(x_train, y_train)

        print("After HyperParameter Tuning :- ")
        print("Training Score :- ", round(etc_clf.score(x_train, y_train), 4))
        print("Testing Score :- ", round(etc_clf.score(x_test, y_test), 4))
        print()
        print("Now the model looks more Generalized !!!")
        return etc_clf

    def rfctuner(self):
        x_train ,x_test, y_train, y_test = self.data.splitthedata()
        rfc = self.model.rfc()
        params = {
            'criterion' : ['gini', 'entropy'],
            'max_depth' : [i for i in range(1,10,1)],
            'max_features': ['sqrt', 'auto', 'log2', None],
            'n_estimators': [i for i in range(100, 3000, 30)],
            'min_samples_split': [i for i in range(2, 50, 1)],
            'min_samples_leaf': [i for i in range(1, 50, 1)]
        }
        randomcv_rfc = RandomizedSearchCV(rfc, param_distributions=params, n_jobs=-1, cv=5, n_iter=100, verbose=True)
        randomcv_rfc.fit(x_train, y_train)
        rfc_best = randomcv_rfc.best_params_
        rfc_clf = RandomForestClassifier(n_estimators=rfc_best['n_estimators'],
                                         min_samples_leaf=rfc_best['min_samples_leaf'],
                                         min_samples_split=rfc_best['min_samples_split'],
                                         max_features=rfc_best['max_features'],
                                         criterion=rfc_best['criterion'], max_depth=rfc_best['max_depth'])
        rfc_clf.fit(x_train, y_train)

        print("After HyperParameter Tuning :- ")
        print("Training Score :- ", round(rfc_clf.score(x_train, y_train), 4))
        print("Testing Score :- ", round(rfc_clf.score(x_test, y_test), 4))
        print()
        print("Now the model looks more Generalized !!!")
        return rfc_clf


