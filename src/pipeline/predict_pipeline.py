import sys
import pandas as pd
import numpy as np
from src.exception_config import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass 

    def predict(self,features):
        try:
            model_path = 'artifacts\model.csv'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 carlength: float,
                 carwidth: float,
                 curbweight: float,
                 enginesize: float,
                 horsepower: float,
                 citympg: float,
                 highwaympg: float,
                 CarName: str,
                 drivewheel: str,
                 enginelocation: str,
                 fuelsystem: str):
        
        self.carlength = carlength
        self.carwidth = carwidth
        self.curbweight = curbweight
        self.enginesize = enginesize
        self.horsepower = horsepower
        self.citympg = citympg
        self.highwaympg = highwaympg
        self.CarName = CarName
        self.drivewheel = drivewheel
        self.enginelocation = enginelocation
        self.fuelsystem = fuelsystem

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "carlength": [self.carlength],
                "carwidth": [self.carwidth],
                "curbweight": [self.curbweight],
                "enginesize": [self.enginesize],
                "horsepower": [self.horsepower],
                "citympg": [self.citympg],
                "highwaympg": [self.highwaympg],
                "CarName": [self.CarName],
                "drivewheel": [self.drivewheel],
                "enginelocation": [self.enginelocation],
                "fuelsystem": [self.fuelsystem]
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)


