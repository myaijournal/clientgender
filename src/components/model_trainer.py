import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_resampled_path, y_train_resampled_path, X_test_path, y_test_path):
        try:
            X_train, y_train, X_test, y_test = [pd.read_csv(path) for path in ['artifacts\X_train_resampled.csv', 'artifacts\y_train_resampled.csv', 'artifacts\X_test.csv', 'artifacts\y_test.csv']]
            
            X_train = X_train.values
            y_train = y_train.values.ravel()  # Ensuring target is 1-dimensional
            X_test = X_test.values
            y_test = y_test.values.ravel()

            models = { 
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBRegressor": XGBClassifier(),
                "CatBoosting Regressor": CatBoostClassifier(verbose=False),
                "AdaBoost Regressor": AdaBoostClassifier()
           }
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            

        except Exception as e:
            raise CustomException(e,sys)

# with hyper param tuning initiate_model_trainer() will get modeified to

# def initiate_model_trainer(self, X_train_resampled_path, y_train_resampled_path, X_test_path, y_test_path):
#         try:
#             X_train, y_train, X_test, y_test = [pd.read_csv(path) for path in ['artifacts\X_train_resampled.csv', 'artifacts\y_train_resampled.csv', 'artifacts\X_test.csv', 'artifacts\y_test.csv']]
            
#             X_train = X_train.values
#             y_train = y_train.values.ravel()  # Ensuring target is 1-dimensional
#             X_test = X_test.values
#             y_test = y_test.values.ravel()

#             models = { 
#                 "Random Forest": RandomForestClassifier(),
#                 "Gradient Boosting": GradientBoostingClassifier(),
#                 "Decision Tree": DecisionTreeClassifier(),
#                 "XGBRegressor": XGBClassifier(),
#                 "CatBoosting Regressor": CatBoostClassifier(verbose=False),
#                 "AdaBoost Regressor": AdaBoostClassifier()
#            }
            
#             param_grids = {
#                 "Random Forest": {
#                     'n_estimators': [100, 200, 300],
#                     'max_depth': [None, 10, 20, 30],
#                     'min_samples_split': [2, 5, 10],
#                     'min_samples_leaf': [1, 2, 4],
#                     'bootstrap': [True, False]
#                 },
#                 "Gradient Boosting": {
#                     'n_estimators': [100, 200, 300],
#                     'learning_rate': [0.01, 0.1, 0.2],
#                     'max_depth': [3, 4, 5],
#                     'subsample': [0.8, 1.0],
#                     'min_samples_split': [2, 5, 10],
#                     'min_samples_leaf': [1, 2, 4]
#                 },
#                 "Decision Tree": {
#                     'max_depth': [None, 10, 20, 30],
#                     'min_samples_split': [2, 5, 10],
#                     'min_samples_leaf': [1, 2, 4],
#                     'criterion': ['gini', 'entropy']
#                 },
#                 "XGBClassifier": {
#                     'n_estimators': [100, 200, 300],
#                     'learning_rate': [0.01, 0.1, 0.2],
#                     'max_depth': [3, 4, 5],
#                     'subsample': [0.8, 1.0],
#                     'colsample_bytree': [0.8, 1.0],
#                     'gamma': [0, 0.1, 0.2]
#                 },
#                 "CatBoostClassifier": {
#                     'iterations': [100, 200, 300],
#                     'learning_rate': [0.01, 0.1, 0.2],
#                     'depth': [3, 4, 5],
#                     'l2_leaf_reg': [1, 3, 5],
#                     'border_count': [32, 64, 128]
#                 },
#                 "AdaBoostClassifier": {
#                     'n_estimators': [50, 100, 200],
#                     'learning_rate': [0.01, 0.1, 1.0]
#                 }
#             }
            
#             model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,
#                                              models=models, param_grids = param_grids)
            
#             ## To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             ## To get best model name from dict

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score<0.6:
#                 raise CustomException("No best model found")
#             logging.info(f"Best found model on both training and testing dataset")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted=best_model.predict(X_test)

#             accuracy = accuracy_score(y_test, predicted)
#             return accuracy
            

#         except Exception as e:
#             raise CustomException(e,sys)