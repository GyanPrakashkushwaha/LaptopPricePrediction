
import os
import sys
import numpy as np

from sklearn.preprocessing import RobustScaler
from src.Components.Data_Classes import DataTransformationConfig
import pandas as pd
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.utils import MainUtils


class DataTransformation:
    def __init__(self,train_data_path,test_data_path):

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path        

        self.data_transformation_config = DataTransformationConfig()

        self.utils = MainUtils()

        test_df = pd.read_csv(self.test_data_path)
        train_df = pd.read_csv(self.train_data_path)
        self.dataframe = pd.concat([train_df,test_df])

    def imputer_scaler(self):
        try:
            imputer = ('imputer',SimpleImputer(strategy='constant',fill_value=0))
            scaler = ('scaler',RobustScaler())

            preprocessor = Pipeline(steps=[
                imputer,
                scaler
            ])

            return preprocessor
        except Exception as e :
            raise CustomException(e,sys)
        

    def data_encoding(self):
        

        
    def initiate_data_transformation(self):

        logging.info(
            "Entered initiate_data_transformation method of Data_Transformation class"
        )
      
        try:
           dataframe = self.dataframe
           X = dataframe.drop(columns='Price')
           y = dataframe['Price']

           X_train, X_test, y_train, y_test = train_test_split(
                      X, y, test_size=0.33, random_state=42)
           
           prerocessor = self.imputer_scaler()
           

           X_train_scaled = prerocessor.fit_transform(X_train)
           X_test_scaled = prerocessor.transform(X_test)

           preprocessor_path = self.data_transformation_config.transformed_object_file_path

           os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)

           self.utils.save_obj(file_path=preprocessor_path,obj=prerocessor)

           train_arr = np.c_[X_train_scaled,np.array(y_train)]
           test_arr = np.c_[X_test_scaled,np.array(y_test)]

           return(
               train_arr,
               test_arr,
               preprocessor_path
           )
           
        except Exception as e :
            raise CustomException(e,sys)
        
