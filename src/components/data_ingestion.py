import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_data=pd.read_csv('C:\\Users\\shant\\OneDrive\\Desktop\\mlproject\\train.csv')
            logging.info("Train data is loaded")

            test_data=pd.read_csv('C:\\Users\\shant\\OneDrive\\Desktop\\mlproject\\test.csv')
            logging.info("Test data is loaded")

            train_data.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,header=True,index=False)
            logging.info("Data ingestion is completed")
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)    
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

