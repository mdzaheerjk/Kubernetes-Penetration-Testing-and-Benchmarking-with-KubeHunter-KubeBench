import os
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, input_path: str = "artifacts", output_path: str = "artifacts"):
        self.input_path = input_path
        self.output_path = output_path
        self.model = xgb.XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        os.makedirs(self.output_path, exist_ok=True)
        logger.info(f"ModelTraining initialized with input: {self.input_path}, output: {self.output_path}")

    def load_data(self):
        try:
            self.x_train = joblib.load(os.path.join(self.input_path, "x_train.pkl"))
            self.x_test = joblib.load(os.path.join(self.input_path, "x_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.input_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.input_path, "y_test.pkl"))
            logger.info("Data loaded successfully for model training.")
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data for training", e)

    def train_model(self):
        try:
            logger.info("Model training started...")
            self.model.fit(self.x_train, self.y_train)

            model_save_path = os.path.join(self.output_path, "model.pkl")
            joblib.dump(self.model, model_save_path)
            logger.info(f"Trained model saved to {model_save_path}")
        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException("Failed to train model", e)

    def eval_model(self):
        try:
            train_score = self.model.score(self.x_train, self.y_train)
            logger.info(f"Training Model Score: {train_score:.4f}")

            y_pred = self.model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

            logger.info(f"Test Accuracy:  {accuracy:.4f}")
            logger.info(f"Test Precision: {precision:.4f}")
            logger.info(f"Test Recall:    {recall:.4f}")
            logger.info(f"Test F1-Score:  {f1:.4f}")
        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)

    def run(self):
        self.load_data()
        self.train_model()
        self.eval_model()
        logger.info("Model training and evaluation pipeline completed successfully.")


if __name__ == "__main__":
    trainer = ModelTraining(input_path="artifacts", output_path="artifacts")
    trainer.run()