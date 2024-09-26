# model_trainer.py
import hydra
import logging
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error)
import numpy as np    
from omegaconf import DictConfig
from data_loader import DataLoader  # Import the DataLoader class
import joblib 
import sys
from pathlib import Path
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)

class ModelTrainer:
    def __init__(self, data_loader,cfg: DictConfig):
        """
        Initializes the ModelTrainer with a DataLoader instance.
        
        Parameters:
        - data_loader: DataLoader: The DataLoader instance with the loaded data.
        - cfg: DictConfig: The Hydra configuration object for the pipeline.
        """
        self.data_loader = data_loader
        file_path = self.data_loader.current_dir / cfg.pipeline.model_path 
        self.model_path = str(file_path)
        logging.info(f"ModelTrainer initialized. Model will be saved to {self.model_path}")

    def train(self):
        """
        Trains the model using the training data.
        """
        try:
            X_train, y_train = self.data_loader.get_train_data()
            self.data_loader.pipeline.fit(X_train, y_train)
            logging.info("Model trained successfully.")
            self.save_model()
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def save_model(self):
        """
        Saves the trained model to a file.
        """
        try:
            model_path = os.path.join(self.model_path, 'trained_model.pkl')
            joblib.dump(self.data_loader.pipeline, model_path)
            logging.info(f"Model saved successfully at: {model_path}")
        except IOError as e:
            logging.error(f"Error saving the model: {e}")
            raise    

    def print_metrics(self, predictions, target):
        """
        Prints evaluation metrics based on the predictions and true targets.
        
        Parameters:
        - predictions: np.ndarray: The predicted values.
        - target: np.ndarray: The true values.
        """
        try:
            rmse = np.sqrt(mean_squared_error(predictions, target))
            mape = mean_absolute_percentage_error(predictions, target)
            mae = mean_absolute_error(predictions, target)
            
            logging.info(f"RMSE: {rmse}")
            logging.info(f"MAPE: {mape}")
            logging.info(f"MAE: {mae}")
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            raise

    def evaluate(self):
        """
        Evaluates the model using the test data.
        """
        try:
            X_test, y_test = self.data_loader.get_test_data()
            predictions = self.data_loader.pipeline.predict(X_test)
            logging.info("Model evaluation completed.")
            self.print_metrics(predictions, y_test)
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise

       

#Using Hydra for configuration files and automate experiments
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    """
    Main function that loads data, trains the model, and evaluates it.
    
    Parameters:
    - cfg: DictConfig: The Hydra configuration object.
    """
    try:
        data_loader = DataLoader(cfg)

        if cfg.database.enabled:
            data_loader.connect_to_database(cfg.database)
        else:    
            if not data_loader.load_data():
                sys.exit(0)

            data_loader.prepare_data()

            model_trainer = ModelTrainer(data_loader, cfg)
            model_trainer.train()
            model_trainer.evaluate()
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

