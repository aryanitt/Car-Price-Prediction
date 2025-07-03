import os
import sys
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from src.exception_config import CustomException
from src.logger_config import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_array , test_array):
        try:
            logging.info("Splitting independent and dependent features")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],train_array[:,-1],
                test_array[:,:-1], test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGB Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),

            }

            param = {
                "Linear Regression": {
                    "fit_intercept": [True, False]
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 10],
                    "weights": ["uniform", "distance"],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                },
                "Random Forest Regressor": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "max_features": ["auto", "sqrt"],
                },
                "XGB Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "reg_lambda": [1, 2]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1]
                },
                "CatBoost Regressor": {
                    "iterations": [200, 500],
                    "learning_rate": [0.01, 0.05],
                }
            }

            model_report:dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                param=param

            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with acceptable R² score")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            score = r2_score(y_test, predicted)
            return score

        except Exception as e:
            raise CustomException(e,sys)



