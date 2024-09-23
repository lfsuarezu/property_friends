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

class ModelTrainer:
    def __init__(self, data_loader,cfg: DictConfig):
        """Initializes the ModelTrainer with a DataLoader instance."""
        self.data_loader = data_loader

        # Get the current working directory
        file_path = self.data_loader.current_dir.parent / cfg.pipeline.model_path 
        self.model_path = str(file_path)

    def train(self):
        """Trains the model using the training data."""
        X_train, y_train = self.data_loader.get_train_data()
        self.data_loader.pipeline.fit(X_train, y_train)
        print("Model trained successfully.")
        self.save_model()

    def save_model(self):
        """Saves the trained model to a file."""       
        model_path=os.path.join(self.model_path, 'trained_model.pkl')
        joblib.dump(self.data_loader.pipeline, model_path)
        #logging.info("Model saved successfully at: %s", model_path)    

    def print_metrics(self, predictions, target):
        print("RMSE: ", np.sqrt(mean_squared_error(predictions, target)))
        print("MAPE: ", mean_absolute_percentage_error(predictions, target))
        print("MAE : ", mean_absolute_error(predictions, target))

    def evaluate(self):
        """Evaluates the model using the test data."""
        X_test, y_test = self.data_loader.get_test_data()
        predictions = self.data_loader.pipeline.predict(X_test)
        
        # You can add your own evaluation metric here
        self.print_metrics(predictions,y_test)


# Example usage with Hydra

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    data_loader = DataLoader(cfg)
    if not data_loader.load_data():
        sys.exit(0)    
    data_loader.prepare_data()

    model_trainer = ModelTrainer(data_loader,cfg)
    model_trainer.train()
    model_trainer.evaluate()

if __name__ == "__main__":
    main()

