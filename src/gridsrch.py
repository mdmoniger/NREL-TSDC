import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
from imblearn.combine import SMOTEENN

class gridsrch():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def gridsrch(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        # if model == 'rfc':
        rfc = RandomForestClassifier()
        rfcgrid = GridSearchCV(rfc, param_grid={
                'n_estimators':[10, 50, 100, 200, 500],
                'max_features':[None, 1, 2, 3, 4, 5, 6, 7, 8],
                'max_depth':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                'class_weight':[None, 'balanced_subsample', 'balanced']},
                scoring='f1', n_jobs=-1, refit=True)
        print('fitting rfc')
        rfc = rfcgrid.fit(X_train, y_train)
        rfcbestpars = rfcgrid.best_params_
        print('pickling rfc')
        pickle.dump(rfc, open('rfc.pkl', 'wb'))
        pickle.dump(rfcbestpars, open('rfcbestpars', 'wb'))
        # if model == 'abc':
        abc = AdaBoostClassifier()
        abcgrid = GridSearchCV(abc, param_grid={
                'n_estimators':[10, 50, 75, 100, 150, 200, 500],
                'learning_rate':[.01, .05, .1, 0.5, 1, 2]},
                scoring='f1', n_jobs=-1, refit=True)
        print('fitting abc')
        abc = abcgrid.fit(X_train, y_train)
        abcbestpars = abcgrid.best_params_
        print('pickling abc')
        pickle.dump(abc, open('abc.pkl', 'wb'))
        pickle.dump(abcbestpars, open('abcbestpars', 'wb'))
        # if model == 'gbc':
        gbc = GradientBoostingClassifier()
        gbcgrid = GridSearchCV(gbc, param_grid=
                {'n_estimators':[50, 75, 100, 150, 200, 500],
                'max_depth':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                'learning_rate':[.01, .05, .1, 0.5, 1, 2]},
                scoring='f1', n_jobs=-1, refit=True)
        print('fitting gbc')
        gbc = gbcgrid.fit(X_train, y_train)
        gbcbestpars = gbcgrid.best_params_
        print('pickling gbc')
        pickle.dump(gbc, open('gbc.pkl', 'wb'))
        pickle.dump(gbcbestpars, open('gbcbestpars', 'wb'))
        return gbc, abc, rfc, rfcbestpars, gbcbestpars, abcbestpars


if __name__ == "__main__":
    with open('../data/X', 'rb') as file:
        X = pickle.load(file)
    with open('../data/ymap', 'rb') as file:
        y = pickle.load(file)
    test = gridsrch(X, y).gridsrch()
