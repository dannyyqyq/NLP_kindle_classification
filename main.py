from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
import sys
from src.exception import CustomException

if __name__ == "__main__":
    try:
        data_ingestion_obj = DataIngestion()
        data_ingestion_config = DataIngestionConfig()
        data_transformation_obj = DataTransformation()
        data_transformation_config = DataTransformationConfig()
        model_trainer_obj = ModelTrainer()

        df, _ = data_ingestion_obj.initiate_data_ingestion()
        (
            train_set,
            test_set,
            _,
            _,
        ) = data_transformation_obj.initiate_data_transformation(df)
        model_trainer_obj.initiate_model_trainer(train_set, test_set)

    except Exception as e:
        raise CustomException(e, sys)
