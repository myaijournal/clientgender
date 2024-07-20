import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    clientgender_data_path: str = os.path.join("artifacts", "clientgender.csv")
    purchase_history_data_path: str = os.path.join("artifacts", "purchase_history.csv")
    clientgender_rawdata_path: str = os.path.join("artifacts", "clientgender_rawdata.csv")
    purchase_history_rawdata_path: str = os.path.join("artifacts", "purchase_history_rawdata.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df1=pd.read_csv('notebook\data\ClientGender.csv')
            logging.info('Read the clientgender dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.clientgender_data_path),exist_ok=True)

            df1.to_csv(self.ingestion_config.clientgender_rawdata_path,index=False,header=True)

            df2=pd.read_csv('notebook\data\PurchaseHistory.csv')
            logging.info('Read the purchase history dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.purchase_history_data_path),exist_ok=True)

            df2.to_csv(self.ingestion_config.purchase_history_rawdata_path,index=False,header=True)

            logging.info("Ingestion of both the datas is completed")

            return(
                self.ingestion_config.clientgender_data_path,
                self.ingestion_config.purchase_history_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()



