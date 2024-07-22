import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import os



@dataclass
class DataPreparationConfig:
    merged_data_path: str=os.path.join('artifacts',"merged_data.csv")
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    
class DataPreparation:
    def __init__(self):
        self.data_preparation_config=DataPreparationConfig()
        
    def merge_data_objects(self, clientgender_rawdata_path, purchase_history_rawdata_path):
        '''
        This function is responsible for converting/correcting data types and then merging two datasets
        
        '''
        try:

            df1 = pd.read_csv(clientgender_rawdata_path)
            df2 = pd.read_csv(purchase_history_rawdata_path)

            logging.info("Read the datasets for preparation")

            df1['client_id'] = pd.to_numeric(df1['client_id'], errors='coerce')
            df1 = df1.dropna(subset=['client_id'])
            df1.reset_index(drop=True, inplace=True)

            logging.info("Converted client_id to numeric in clientgender dataset")

            df2['client_id'] = pd.to_numeric(df2['client_id'], errors='coerce')
            df2 = df2.dropna(subset=['client_id'])
            df2.reset_index(drop=True, inplace=True)

            logging.info("Converted client_id to numeric in purchase_history dataset")

            merged_df = pd.merge(df1, df2, on='client_id', how='inner')

            logging.info("Merged two cleaned datasets")

            merged_df['purchase_amount'] = pd.to_numeric(merged_df['purchase_amount'], errors='coerce')

            merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

            merged_df = merged_df.dropna()

            logging.info("Cleaned date, purchase amount and dropped null values in Merged dataset")

            merged_df.to_csv(self.data_preparation_config.merged_data_path,index=False,header=True)

            logging.info("Saved merged dataset to artifacts")

            return(
                self.data_preparation_config.merged_data_path

            )
  
        except Exception as e:
            raise CustomException(e,sys)
        
