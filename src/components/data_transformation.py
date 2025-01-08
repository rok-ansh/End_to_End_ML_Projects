import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
# This is use to create the pipeline 
from sklearn.compose import ColumnTransformer
# For missing value 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import saved_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data Transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Creating the pipeline which needs to run on the training Dataset
            num_pipeline = Pipeline(
                steps = [
                    # Handling the missing value 
                    ("imputer", SimpleImputer(strategy="median")),
                    # Doing the Standard Scaling 
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical Columns Scaling Completed : {numerical_columns}")


            # Creating Categorical Pipeline 
            cat_pipeline = Pipeline(
                steps =[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
                    ("scaler", StandardScaler())

                ]
            )
            logging.info(f"Categorical Columns Encoding Completed : {categorical_columns}")


            # To combine the num_pipeline and cat_pipeline we will ColumnTransformer 
            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)   
                ]
            )
            logging.info("Column Transformer completed for Cat_pipeline and Num_pipeline")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    # The train_path and test_path we are getting it from data ingestion 
    def intiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining  Preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_column = ["writing_score", "reading_score"]

            # Getting the target column separated both in traina and test Data 
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")
            # Used for saving the pickle file 
            saved_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
