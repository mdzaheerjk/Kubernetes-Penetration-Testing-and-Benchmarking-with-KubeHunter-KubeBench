import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, input_path: str = "Data/Data.csv", output_path: str = "artifacts"):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

        os.makedirs(self.output_path, exist_ok=True)
        logger.info(f"DataProcessing initialized with input: {self.input_path}, output: {self.output_path}")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            self.df.columns = self.df.columns.str.strip()
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", e)

    def preprocess(self):
        try:
            if "Date" not in self.df.columns:
                raise ValueError(f"'Date' column not found. Available columns: {self.df.columns.tolist()}")

            # Convert Date to datetime and extract date features
            self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
            self.df["Year"] = self.df["Date"].dt.year
            self.df["Month"] = self.df["Date"].dt.month
            self.df["Day"] = self.df["Date"].dt.day
            self.df.drop("Date", axis=1, inplace=True)

            # Fill numerical missing values with mean
            numerical = self.df.select_dtypes(include=["int64", "float64"]).columns
            for col in numerical:
                self.df[col] = self.df[col].fillna(self.df[col].mean())

            # Drop remaining missing values
            self.df.dropna(inplace=True)
            logger.info("Basic data preprocessing and imputation completed.")
        except Exception as e:
            logger.error(f"Error while preprocessing data: {e}")
            raise CustomException("Failed to preprocess data", e)

    def label_encode(self):
        try:
            categorical = [
                "Location",
                "WindGustDir",
                "WindDir9am",
                "WindDir3pm",
                "RainToday",
                "RainTomorrow"
            ]

            encoders = {}
            for col in categorical:
                if col in self.df.columns:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    encoders[col] = le

            joblib.dump(encoders, os.path.join(self.output_path, "encoders.pkl"))
            logger.info("Categorical features label encoded and encoders saved.")
        except Exception as e:
            logger.error(f"Error while label encoding data: {e}")
            raise CustomException("Failed to label encode data", e)

    def split_data(self):
        try:
            x = self.df.drop("RainTomorrow", axis=1)
            y = self.df["RainTomorrow"]

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            joblib.dump(x_train, os.path.join(self.output_path, "x_train.pkl"))
            joblib.dump(x_test, os.path.join(self.output_path, "x_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path, "y_test.pkl"))

            logger.info(f"Data split saved to {self.output_path}. Train: {x_train.shape}, Test: {x_test.shape}")
        except Exception as e:
            logger.error(f"Error while splitting data: {e}")
            raise CustomException("Failed to split data", e)

    def run(self):
        try:
            self.load_data()
            self.preprocess()
            self.label_encode()
            self.split_data()
            logger.info("DataProcessing pipeline executed successfully.")
        except Exception as e:
            logger.error(f"Data processing pipeline failed: {e}")
            raise


if __name__ == "__main__":
    processor = DataProcessing(input_path="Data/Data.csv", output_path="artifacts")
    processor.run()