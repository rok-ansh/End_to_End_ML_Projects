import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    # Output will save in this path 
    # The below 3 are the input which i am giving to dataIngestion component anf now dataIngestion know where to save the train path, test path and raw path
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data Ingestion method or component")
        try:
            # Read the dataset 
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe.")

            # Creating a aritfacts folder for Train data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # saving the raw data in the  artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test Split initiated")
            # Splitting the data into train and test 
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # saving the train_set to artifacts folder 
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Saving the test_set to artifacts folder 
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )

        except Exception as e :
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
