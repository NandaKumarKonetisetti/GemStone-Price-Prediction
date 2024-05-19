import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from src.exception import CustomException
from src.logger  import logging

from dataclasses import dataclass
from src.utils import evaluate_models, save_object

import os 
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str =   os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting the dependent and Independent variables from train and test")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "LinearRegress":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "ElasticNet":ElasticNet()
            }
            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models)
            logging.info("*"*65)
            logging.info(f"Model report :{model_report}")
            # To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            logging.info(f"Best Model Found :{best_model_name},R2 Score :{best_model_score}")
            print(f"Best Model Found :{best_model_name},R2 Score :{best_model_score}")
        
        except Exception as e:
            logging.error("Exception occured in model training",e,sys)
            raise CustomException(e,sys)