

import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor ,
                                GradientBoostingRegressor,
                                VotingRegressor,
                                StackingRegressor)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error , r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.exception import CustomException
import warnings
warnings.filterwarnings('ignore')
from src.logger import logging
import pandas as pd


class ModelTrainer:

    def __init__(self,train_arr,test_arr):
        
        self.models = {
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
            }

        self.params = {
            "Random Forest": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                # 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'criterion': ['squared_error', 'friedman_mse'],
                # 'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "XGBRegressor": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "CatBoosting Regressor": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost Regressor": {
                'learning_rate': [0.1, 0.01, 0.5, 0.001],
                # 'loss': ['linear', 'square', 'exponential'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        self.train_arr = train_arr
        self.test_arr = test_arr
        # self.preprocessor = preprocessor_file

        self.X_train, self.X_test, self.y_train,self.y_test = (
            train_arr[:,[0,1,2,3,5,6,7,8,9,10,11,12]],
            test_arr[:,[0,1,2,3,5,6,7,8,9,10,11,12]],
            train_arr[:,4],
            test_arr[:,4]
        )

        self.model_list = []
        self.accuracy_list = []

    def evaluate_model(self,true,pred):
        r2_score_model = r2_score(y_true=true,y_pred=pred)
        return r2_score_model
    

    def initiate_model_training(self):
        models = self.models
        params = self.params
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        try:
            for i, model_name in enumerate(self.models.keys()):
                model = models[model_name]
                param = params[model_name]

                grid_search_cv = GridSearchCV(estimator=model, param_grid=param, cv=5)
                grid_search_cv.fit(X_train, y_train)

                best_params = grid_search_cv.best_params_
                model.set_params(**best_params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                r2_score_model = self.evaluate_model(true=y_test, pred=y_pred)

                self.model_list.append(model_name)
                self.accuracy_list.append(r2_score_model)

            logging.info(f'the best and and its list is{pd.DataFrame(list(zip(list(self.models.keys()),self.accuracy_list)),columns=["model","accuracy"])}')

            best_model = max(list(zip(self.model_list, self.accuracy_list)))
            

            

            return print(list(zip(self.model_list, self.accuracy_list)))
                    

        except Exception as e:
            raise CustomException(e, sys)
    




            


           

