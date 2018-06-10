import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import pickle

def predict_transit(self, model, X, ymap):
    X_train, X_test, y_train, y_test = train_test_split(X, ymap)
    # model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return precision_score(y_test, y_pred), recall_score(y_test, y_pred)


if __name__ == "__main__":
    with open('abc.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('../data/X', 'rb') as file:
        X = pickle.load(file)
    with open('../data/ymap', 'rb') as file:
        ymap = pickle.load(file)
    prec, rec = predict_transit(df).predict_transit(model, X, ymap)
