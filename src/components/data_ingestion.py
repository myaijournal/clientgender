import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.components.data_preparation import DataPreparation
from src.components.data_preparation import DataPreparationConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
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

            os.makedirs(os.path.dirname(self.ingestion_config.clientgender_rawdata_path),exist_ok=True)

            df1.to_csv(self.ingestion_config.clientgender_rawdata_path,index=False,header=True)

            df2=pd.read_csv('notebook\data\PurchaseHistory.csv')
            logging.info('Read the purchase history dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.purchase_history_rawdata_path),exist_ok=True)

            df2.to_csv(self.ingestion_config.purchase_history_rawdata_path,index=False,header=True)

            logging.info("Ingestion of both the datas is completed")

            return(
                self.ingestion_config.clientgender_rawdata_path,
                self.ingestion_config.purchase_history_rawdata_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    
    clientgender_data, purchase_history_data = obj.initiate_data_ingestion()

    data_prep = DataPreparation()

    merged_data_path = data_prep.merge_data_objects(clientgender_data, purchase_history_data)

    data_trans = DataTransformation()

    feature_engineered_data_path = data_trans.create_features(merged_data_path)

    features_path, target_path, preprocessor_obj_path = data_trans.apply_preprocessing(feature_engineered_data_path)

    X_train_path, X_test_path, y_train_path, y_test_path, X_train_resampled_path, y_train_resampled_path = data_trans.split_and_resample(features_path, target_path)

    model_training = ModelTrainer()

    print(model_training.initiate_model_trainer(X_train_path, X_test_path, y_train_path, y_test_path))






    






