import os
import sys
import pandas as pd
import numpy as np
from src.logger_config import logging
from src.exception_config import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_config(self):
        try:
            numerical_features = ['carlength', 'carwidth', 'curbweight', 
                                  'enginesize', 'horsepower', 'citympg', 'highwaympg']
            
            cat_features = ['CarName', 'drivewheel', 'enginelocation', 'fuelsystem']
            
            num_pipline = Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns scaling pipeline created")
            logging.info("Categorical columns encoding pipeline created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipline" , num_pipline,numerical_features),
                    ("cat_pipline",cat_pipline,cat_features)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_config()

            target_column_name = "price"

            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_features_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_features_test_df = test_df[target_column_name]

            print("▶ input_features_train_df type:", type(input_features_train_df))
            print("▶ input_features_train_df shape:", input_features_train_df.shape)
            print("▶ Columns:", input_features_train_df.columns.tolist())
            print("▶ Head:")
            print(input_features_train_df.head())


            

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            print("Train shape (input):", input_features_train_arr.shape)
            print("Train shape (target):", np.array(target_features_train_df).shape)

            if hasattr(input_features_train_arr,"toarray"):
                input_features_train_arr = input_features_train_arr.toarray()
                input_features_test_arr = input_features_test_arr.toarray()

            train_arr = np.c_[
                input_features_train_arr, np.array(target_features_train_df)
            ]

            test_arr = np.c_[
                input_features_test_arr, np.array(target_features_test_df)
            ]

            
            print(train_arr.shape)
            print(test_arr.shape)


            save_object(
                file_path=self.transformation_config.preprocessor_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)







        
            
