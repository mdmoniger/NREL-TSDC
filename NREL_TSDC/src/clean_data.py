import pandas as pd
from imblearn.combine import SMOTEENN
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

class clean_data:
    def __init__(self, df=None):
        self.df = df
    def get_percentage_missing(self):
        ''' Calculates percentage of NaN values in DataFrame
        :param self.df: Pandas DataFrame object
        :return: float
        '''
        num = self.df.isnull().sum()
        den = len(series)
        return round(num/den, 7)
    def clean_data(self):
        #drop rows that having no data in target column
        self.df.dropna(subset=[target], inplace=True)
        for col in self.df:
            self.df[col].fillna((self.df[col].mean()), inplace=True)
    def map_and_split(self):
        X = self.df[features]
        y = self.df[target]
        X, y = SMOTEENN().fit_sample(X, y)
        dict_not_transit = dict((key, 0) for key in range(15))
        dict_transit = dict((key, 1) for key in range(15,30))
        mode_dict = {**dict_not_transit, **dict_transit}
        ymap = y.map(mode_dict)
        X, ymap = SMOTEENN().fit_sample(X, ymap)
        # pickle.dump(X, open('X', 'wb'))
        # pickle.dump(ymap, open('ymap', 'wb'))
