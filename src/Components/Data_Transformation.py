
import os
import sys
import numpy as np
from sklearn.discriminant_analysis import StandardScaler

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from src.Components.Data_Classes import DataTransformationConfig
import pandas as pd
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
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
            imputer = ('imputer',SimpleImputer())
            scaler = ('scaler',RobustScaler())

            impute_scale = Pipeline(steps=[
                imputer,
                scaler])

            return impute_scale
        except Exception as e :
            raise CustomException(e,sys)
        

    def data_encoding(self):
        df = self.dataframe
        X = df.drop(columns=['Price'])
        y = df['Price']

        impute_scale = self.imputer_scaler()

        cat_cols = [features for features in X.columns if X[features].dtypes == 'O']
        num_cols = [features for features in X.columns if X[features].dtypes != 'O']

        cat_pipeline = Pipeline(
            steps=[
                ('ohe',OneHotEncoder(sparse_output=False, # this will return me numpy array by the first time
                drop='first')),
                ('imputer',SimpleImputer())])
                
        num_pipeline = Pipeline(steps=[
                ('scaler',StandardScaler())])

        encoding_obj = ColumnTransformer(transformers=[
            ('cat_col_pileline',cat_pipeline,cat_cols),
            ('num_col_pipeline',num_pipeline,num_cols),
            ('impute_scale_pipeline_cat_col',impute_scale,num_cols)
        ],remainder='passthrough')

        return encoding_obj

        
    def initiate_data_transformation(self):

        logging.info(
            "Entered initiate_data_transformation method of Data_Transformation class")
      
        try:
           dataframe = self.dataframe
           X = dataframe.drop(columns='Price')
           y = dataframe['Price']

           X_train, X_test, y_train, y_test = train_test_split(
                      X, y, test_size=0.33, random_state=42)
           

           preprocessor = self.data_encoding()

           X_train_scaled = preprocessor.fit_transform(X_train)
           X_test_scaled = preprocessor.transform(X_test)

           preprocessor_file_path = self.data_transformation_config.transformed_object_file_path

           os.makedirs(os.path.dirname(preprocessor_file_path),exist_ok=True)

           self.utils.save_obj(file_path=preprocessor_file_path,obj=preprocessor)

           train_arr = np.c_[X_train_scaled,np.array(y_train)]
           test_arr = np.c_[X_test_scaled,np.array(y_test)]

           return(
               train_arr,
               test_arr,
               preprocessor_file_path
           )
           
        except Exception as e :
            raise CustomException(e,sys)
        
