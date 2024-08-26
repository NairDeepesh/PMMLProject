import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# Data class to store file paths for preprocessor objects
@dataclass
class DataTransformationConfig:
    preprocessor_x_obj_file_path: str = os.path.join('artifacts', 'preprocessor_x.pkl')
    preprocessor_y_obj_file_path: str = os.path.join('artifacts', 'preprocessor_y.pkl')

# Log transformation function
def log_transform(x):
    return np.log1p(x)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        

    def get_data_transformer_x(self):
        """
        Retrieves the transformer object for features (X).
        
        Returns:
        ColumnTransformer: The transformer object for feature preprocessing.
        """
        try:
            logging.info("Fetching transformer object for feature preprocessing (X)")
            
            log_transformer = Pipeline(steps=[
            ('log', FunctionTransformer(log_transform)),
            ('scaler', RobustScaler())
             ])
            
            standard_transformer = Pipeline(steps=[
            ('scaler', RobustScaler()),
            ('pca', PCA(n_components=0.95))  # Retain 95% of variance
            ])

            robust_transformer = Pipeline(steps=[
                ('scaler', RobustScaler()),
                ('pca', PCA(n_components=0.95))  # Retain 95% of variance
            ])

            standard_columns = ['T50', 'P30', 'Ps30', 'phi', 'BPR', 'W32']
            log_columns=['time_in_cycles', 'Nc']
            robust_columns=['NRf', 'htBleed']

            preprocessor_x = ColumnTransformer(
                transformers=[
                    ("log", log_transformer,log_columns),
                    ("standard", standard_transformer, standard_columns),
                    ("robust", robust_transformer,robust_columns)
                ]
            )
            return preprocessor_x
        
        except Exception as e:
            raise CustomException(e,sys)

    def get_data_transformer_y(self):
        """
        Retrieves the transformer object for the independent feature (y).
        
        Returns:
        Pipeline: The transformer object for target variable preprocessing.
        """

        try:
            logging.info("Fetching transformer object for independent feature preprocessing (y)")
            
            preprocessor_y = Pipeline(steps=[
                ('log', FunctionTransformer(log_transform)),
                ('scaler', RobustScaler())
            ])
            return preprocessor_y
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df= pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Reading of Train and Test Data Completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj_x=self.get_data_transformer_x()
            preprocessing_obj_y=self.get_data_transformer_y()

            target_column_name='RUL'
            

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[[target_column_name]]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[[target_column_name]]

            logging.info(
                f"Applying preprocessing object on training dataframe and test dataframe"
            )

            input_feature_train_arr=preprocessing_obj_x.fit_transform(input_feature_train_df)
            target_feature_train_arr=preprocessing_obj_y.fit_transform(target_feature_train_df)
            input_feature_test_arr=preprocessing_obj_x.transform(input_feature_test_df)
            target_feature_test_arr=preprocessing_obj_y.transform(target_feature_test_df) 

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr.flatten()]  # Flattening to align dimensions
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr.flatten()]  # Flattening to align dimensions

            logging.info(f"Saving Preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_x_obj_file_path,
                obj=preprocessing_obj_x
            )

            save_object(
                file_path=self.data_transformation_config.preprocessor_y_obj_file_path,
                obj=preprocessing_obj_y
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_x_obj_file_path,
                self.data_transformation_config.preprocessor_y_obj_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)
            


