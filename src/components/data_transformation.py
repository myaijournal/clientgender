import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.utils import save_object

from src.exception import CustomException
from src.logger import logging
import os


@dataclass
class DataTransformationConfig:
    merged_data_path: str = os.path.join('artifacts', "merged_data.csv")
    feature_engineered_data_path: str = os.path.join('artifacts', "feature_engineered_data.csv")
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    features_path: str = os.path.join('artifacts', "features.csv")
    target_path: str = os.path.join('artifacts', "target.csv")
    X_train_path: str = os.path.join('artifacts', "X_train.csv")
    X_test_path: str = os.path.join('artifacts', "X_test.csv")
    y_train_path: str = os.path.join('artifacts', "y_train.csv")
    y_test_path: str = os.path.join('artifacts', "y_test.csv")
    X_train_resampled_path: str = os.path.join('artifacts', "X_train_resampled.csv")
    y_train_resampled_path: str = os.path.join('artifacts', "y_train_resampled.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def create_features(self, merged_data_path):
        '''
        This function is responsible for creating additional features and aggregating purchase behavior over time
        '''
        try:
            merged_df = pd.read_csv(merged_data_path)

            logging.info("Read the merged dataset for feature creation")

            # Create additional features and aggregate purchase behavior
            # 1. Aggregate purchase behavior for each client
            client_features = merged_df.groupby('client_id').agg(
                total_purchase_amount=('purchase_amount', 'sum'),
                average_purchase_amount=('purchase_amount', 'mean'),
                unique_items_purchased=('item_name', 'nunique'),
                number_of_purchases=('purchase_amount', 'count')
            ).reset_index()

            client_features = client_features.merge(
                merged_df[['client_id', 'gender']].drop_duplicates(),
                on='client_id',
                how='left'
            )

            logging.info("Aggregated client features and merged with gender information")

            # 2. Aggregate purchase behavior over time
            merged_df['year_month'] = pd.to_datetime(merged_df['date']).dt.to_period('M')

            monthly_features = merged_df.groupby(['client_id', 'year_month']).agg(
                monthly_total_purchase=('purchase_amount', 'sum'),
                monthly_avg_purchase=('purchase_amount', 'mean'),
                monthly_unique_items=('item_name', 'nunique'),
                monthly_number_of_purchases=('purchase_amount', 'count')
            ).reset_index()

            monthly_features_pivot = monthly_features.pivot_table(
                index='client_id',
                columns='year_month',
                values=['monthly_total_purchase', 'monthly_avg_purchase', 'monthly_unique_items', 'monthly_number_of_purchases']
            )

            monthly_features_pivot.columns = ['_'.join(map(str, col)).strip() for col in monthly_features_pivot.columns.values]
            monthly_features_pivot.reset_index(inplace=True)

            client_features = client_features.merge(monthly_features_pivot, on='client_id', how='left')

            logging.info("Aggregated monthly features and merged with client features")

            # 3. Create item features
            item_features = merged_df.pivot_table(
                index='client_id',
                columns='item_name',
                values='purchase_amount',
                aggfunc='count',
                fill_value=0
            ).reset_index()

            item_features.columns.name = None
            item_features.columns = ['client_id'] + [f'item_count_{col}' for col in item_features.columns[1:]]

            logging.info("Created item features")

            # 4. Aggregate total purchase amount for each item_name
            item_purchase_amounts = merged_df.groupby(['client_id', 'item_name'])['purchase_amount'].sum().unstack(fill_value=0).reset_index()
            item_purchase_amounts.columns = ['client_id'] + [f'purchase_amount_{col}' for col in item_purchase_amounts.columns[1:]]

            logging.info("Aggregated purchase amounts for each item_name")

            # 5. Calculate the percentage of purchase amount for each item_category
            category_purchase_amount = merged_df.groupby(['client_id', 'item_category'])['purchase_amount'].sum().reset_index()

            category_purchase_pivot = category_purchase_amount.pivot_table(
                index='client_id',
                columns='item_category',
                values='purchase_amount',
                aggfunc='sum',
                fill_value=0
            ).reset_index()

            category_purchase_percentages = category_purchase_pivot.drop(columns=['client_id']).div(category_purchase_pivot.drop(columns=['client_id']).sum(axis=1), axis=0) * 100
            category_purchase_percentages.columns = ['% purchase_amount_' + col for col in category_purchase_percentages.columns]
            category_purchase_percentages['client_id'] = category_purchase_pivot['client_id']

            logging.info("Calculated percentage of purchase amounts for each item_category")

            # Merge all features
            client_features = client_features.merge(category_purchase_percentages, on='client_id', how='left')
            client_features = client_features.merge(item_purchase_amounts, on='client_id', how='left')
            client_features = client_features.merge(item_features, on='client_id', how='left')
            client_features.fillna(0, inplace=True)

            logging.info("Merged all features into final dataset")

            # Save the final dataset
            client_features.to_csv(self.data_transformation_config.feature_engineered_data_path, index=False, header=True)

            logging.info("Saved feature engineered dataset to artifacts")

            return self.data_transformation_config.feature_engineered_data_path

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def get_feature_preprocessor(self):
        '''
        This function is responsible for creating a preprocessing pipeline for features
        '''
        try:
            feature_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ]
            )
            return feature_pipeline
        except Exception as e:
            raise CustomException(e, sys)


    def apply_preprocessing(self, feature_engineered_data_path):

        try:
            # Read the data
            data = pd.read_csv(feature_engineered_data_path)
            logging.info("Read feature engineered data")

            # Define target and features
            target_column = 'gender'
            features = data.drop(columns=[target_column])
            target = data[target_column]


            logging.info("Defined target and features")

            # Create preprocessors
            feature_preprocessor = self.get_feature_preprocessor()
            label_encoder = LabelEncoder()
            
             # Apply preprocessing
            processed_features = feature_preprocessor.fit_transform(features)
            target = label_encoder.fit_transform(target)

            logging.info("Applied preprocessing on features and target")

            # Save the preprocessing objects
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={"feature_preprocessor": feature_preprocessor}
            )

            # Save processed data as DataFrames
            processed_features_df = pd.DataFrame(processed_features, columns=features.columns)
            target_df = pd.DataFrame(target, columns=[target_column])

            processed_features_df.to_csv(self.data_transformation_config.features_path, index=False)
            target_df.to_csv(self.data_transformation_config.target_path, index=False)

            logging.info("Saved preprocessed features and target to artifacts")

            return self.data_transformation_config.features_path, self.data_transformation_config.target_path, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def split_and_resample(self, features_path, target_path):
        try:
            # Read the preprocessed features and target
            features = pd.read_csv(features_path)
            target = pd.read_csv(target_path)

            logging.info("Read preprocessed features and target")

            # Split the data using stratified sampling
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, stratify=target, random_state=42
            )

            logging.info("Split the data into training and testing sets using stratified sampling")

            # Save the split data
            X_train.to_csv(self.data_transformation_config.X_train_path, index=False)
            X_test.to_csv(self.data_transformation_config.X_test_path, index=False)
            y_train.to_csv(self.data_transformation_config.y_train_path, index=False)
            y_test.to_csv(self.data_transformation_config.y_test_path, index=False)

            logging.info("Saved the split data to artifacts")

            # Apply SMOTE to the training data
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            logging.info("Applied SMOTE to the training data")

            # Save the resampled data
            X_train_resampled.to_csv(self.data_transformation_config.X_train_resampled_path, index=False)
            y_train_resampled.to_csv(self.data_transformation_config.y_train_resampled_path, index=False)

            logging.info("Saved the resampled training data to artifacts")

            return (
                self.data_transformation_config.X_train_path,
                self.data_transformation_config.X_test_path,
                self.data_transformation_config.y_train_path,
                self.data_transformation_config.y_test_path,
                self.data_transformation_config.X_train_resampled_path,
                self.data_transformation_config.y_train_resampled_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)