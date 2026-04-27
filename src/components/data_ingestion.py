import os
import sys
import pandas as pd
import logging
from src import logger 
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into Data Ingestion method/component")
        try:
            df =pd.read_csv(r"notebook\data\StudentsPerformance.csv")
            logging.info("Read the data to DataFrame")

            logging.info("Creating required folders to contain the files : train,test and raw data")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            logging.info("saving raw dataset into data.csv")
            df.to_csv(self.ingestion_config.raw_data_path)

            logging.info("Initiating Train Test Split")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            logging.info("saving train dataset into train.csv")
            train_set.to_csv(self.ingestion_config.train_data_path)

            logging.info("saving test dataset into test.csv")
            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info("Ingestion of data is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =='__main__':
    train_data,test_data=DataIngestion().initiate_data_ingestion()
    DataTransformation().initiate_data_transformation(train_data,test_data)




