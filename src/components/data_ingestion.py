import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.exception_config import CustomException
from src.logger_config import logging  
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv('notebook/CarPrice_Assignment.csv')  
            logging.info('Read the dataset as dataframe')

            df.drop(columns=['car_ID', 'peakrpm', 'symboling', 'compressionratio', 
                             'stroke', 'carheight','boreratio', 'wheelbase',
                             'cylindernumber', 'doornumber', 'enginetype', 'carbody',
                               'fueltype', 'aspiration'],axis=1,inplace=True)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            print("RAW SHAPE:", df.shape)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            print("Train:", train_set.shape, "Test:", test_set.shape)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is complete")              

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path  
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    
    
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path, test_path)

    obj1 = ModelTrainer()
    print(obj1.initiate_model_trainer(train_arr,test_arr))





   



