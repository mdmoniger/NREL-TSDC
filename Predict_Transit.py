import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
from imblearn import SMOTEENN

class predict_transit:
    def __init__(self, df):
        self.df = df
    def get_percentage_missing(self, df):
       ''' Calculates percentage of NaN values in DataFrame
       :param df: Pandas DataFrame object
       :return: float
       '''
       num = df.isnull().sum()
       den = len(series)
       return round(num/den, 7)
    def clean_data(self, df):
        #drop rows that having no data in target column
        df.dropna(subset=[target], inplace=True)
        for col in df:
            df[col].fillna((df[col].mean()), inplace=True)
    def map_and_split(self, df):
        X = df[features]
        y = df[target]
        X, y = SMOTEENN().fit_sample(X, y)
        dict_not_transit = dict((key, 0) for key in range(15))
        dict_transit = dict((key, 1) for key in range(15,30))
        mode_dict = {**dict_not_transit, **dict_transit}
        ymap = y.map(mode_dict)
        X_train, X_test, y_train, y_test = train_test_split(X, ymap)
    def predict_transit(self, df, model):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return precision_score(y_test, y_pred), recall_score(y_test, y_pred)


if __name__ == "__main__":
    survey_place = pd.read_csv('data/caltrans_full_survey/survey_place.csv')
    columns = ['arr_time', 'dep_time', 'trip_distance_miles', 'air_trip_distance_miles', 'prev_trip_duration_min', 'act_dur', 'act_cnt', 'tract_id', 'county_id', 'state_id', 'mode']
    survey_place_less = survey_place[columns]
    survey_place_less['arr_time'] = pd.to_datetime(survey_place_less['arr_time'], infer_datetime_format=True, format='%H:%M:%S').dt.hour
    survey_place_less['dep_time'] = pd.to_datetime(survey_place_less['dep_time'], infer_datetime_format=True, format='%H:%M:%S').dt.hour
    df = survey_place_less
    target = 'mode'
    features = ['arr_time', 'dep_time', 'trip_distance_miles', 'air_trip_distance_miles', 'prev_trip_duration_min', 'act_dur', 'act_cnt', 'state_id']
    with open('gbc.pkl', 'rb') as file:
        model = pickle.load(file)
