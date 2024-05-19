import os
import sys
from src.components import data_transformation
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation


## Intilize the dataIngestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path :str = os.path.join('artifacts','raw.csv')


#Create a class for DataIngestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method Starts')
        try:
            df = pd.read_csv(os.path.join(os.getcwd(),"notebooks/data/gemstone.csv"))
            logging.info("Dataset read as pandas dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Train Test Split")
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception ocuured at data ingestion class ",e,sys)
            raise CustomException(e,sys)
        

# Run Data Ingestion
if __name__=='__main__':
    obj = DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initate_data_transformation(train_data_path,test_data_path)