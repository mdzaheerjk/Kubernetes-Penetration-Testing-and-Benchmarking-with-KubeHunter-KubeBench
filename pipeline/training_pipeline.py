import os
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from src.logger import get_logger

logger = get_logger("training_pipeline")

def run_pipeline():
    logger.info("Starting complete end-to-end ML training pipeline...")
    
    # 1. Process data and generate train/test splits
    processor = DataProcessing(input_path="Data/Data.csv", output_path="artifacts")
    processor.run()

    # 2. Train model on processed data and evaluate
    trainer = ModelTraining(input_path="artifacts", output_path="artifacts")
    trainer.run()

    logger.info("Training pipeline finished successfully.")

if __name__ == '__main__':
    run_pipeline()