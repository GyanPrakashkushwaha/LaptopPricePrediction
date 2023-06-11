

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
from src.utils import MainUtils
from src.Components.Data_Classes import ModelTrainerConfig
from src.Components.Data_Collection import DataCollection
from src.Components.Data_Transformation import DataTransformation
import warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline








class ModelTrainer:

    def __init__(self,train_arr,test_arr):

        self.utils = MainUtils()
        self.model_trainer_config = ModelTrainerConfig()

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

            best_model_name = max(list(zip(self.model_list, self.accuracy_list)))[0]

            best_model = self.models[best_model_name]

            self.utils.save_obj(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            return (
                best_model
            )
                    
        except Exception as e:
            raise CustomException(e, sys)
        
    
    # def preprocessor_obj(self):
    #     try:
    #         # making pickle file of transformation and model both combined file

    #         self.obj = DataCollection()
    #         self.obj.initiate_data_collection()

    #         model = self.initiate_model_training()

    #         train_df ,test_df=self.obj.initiate_data_collection()

    #         data_trans = DataTransformation(train_data_path=train_df,test_data_path=test_df)

    #         train_arr , test_arr ,_,preprocessor_obj=data_trans.initiate_data_transformation()

    #         self.train_arr = train_arr
    #         self.test_arr = test_arr
    #         # self.preprocessor = preprocessor_file

    #         self.X_train, self.X_test, self.y_train,self.y_test = (
    #             train_arr[:,[0,1,2,3,5,6,7,8,9,10,11,12]],
    #             test_arr[:,[0,1,2,3,5,6,7,8,9,10,11,12]],
    #             train_arr[:,4],
    #             test_arr[:,4]
    #         )


    #         preprocess_model = Pipeline(steps=[
    #             ('preprocessor_obj',preprocessor_obj),
    #             ('model',model)
    #         ])

    #         preprocess_model.fit_transform(self.X_train,self.X_test)

    #         self.utils.save_obj(file_path=self.model_trainer_config.preprocessor_model_file_path,obj=preprocess_model)
    #     #     preprocess_model = Pipeline(
    #     #     steps=[
    #     # ( 'preprocessor_obj',preprocessor_obj),
    #     # ('votingRegressor',votingRegressor)
    #     #     ]
    #     # )
    #     # preprocess_model.fit_transform(X_train,y_train)


    #     except Exception as e:
    #         raise CustomException(e,sys)



if __name__=='__main__':

    obj = ModelTrainer()
    
    




            


           

