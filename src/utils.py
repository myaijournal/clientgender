import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train_resampled, y_train_resampled, X_test, y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train_resampled, y_train_resampled)

            y_train_pred = model.predict(X_train_resampled)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train_resampled, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

## With hyper param tuning above evaluate_models() will become

# def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
#     try:
#         report = {}
        
#         # Dictionary to store the best models from GridSearchCV
#         best_models = {}

#         for name, model in models.items():
            
#             grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, scoring='accuracy')
#             grid_search.fit(X_train, y_train)
            
#             best_model = grid_search.best_estimator_
#             best_models[name] = best_model
            
#             y_train_pred = best_model.predict(X_train)
#             y_test_pred = best_model.predict(X_test)
            
#             train_model_score = accuracy_score(y_train, y_train_pred)
#             test_model_score = accuracy_score(y_test, y_test_pred)
            
#             report[name] = test_model_score

#         return report, best_models
    
#     except Exception as e:
#         raise CustomException(e, sys)
    