import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import make_column_selector as selector
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_loader.log"),
        logging.StreamHandler()
    ]
)

class DataLoader:
    def __init__(self, cfg: DictConfig):
        """
        Initializes the DataLoader with the specified parameters from the config.
        
        Parameters:
        - cfg: DictConfig, the configuration object containing file paths, column specifications,
               and pipeline settings.
        """
        self.train_file = cfg.data_loader.train_file
        self.test_file = cfg.data_loader.test_file
        self.data_path = cfg.data_loader.data_path
        self.target_column = cfg.data_loader.target_column
        self.categorical_cols = cfg.data_loader.categorical_cols
        self.not_consider_column = cfg.data_loader.not_consider_column 
        
        self.train_data = None
        self.test_data = None
        
        # Create the preprocessing pipeline
        self.pipeline = self._create_pipeline(cfg)

        self.current_dir = Path(hydra.utils.get_original_cwd()) 

        logging.info("DataLoader initialized with config: %s", cfg)

    def _create_pipeline(self, cfg):
        """
        Creates a pipeline for preprocessing and modeling.

        Parameters:
        - cfg: DictConfig: The configuration object for the pipeline.
        
        Returns:
        - Pipeline: The constructed pipeline.
        """
        try:
            # Determine the categorical encoder type
            encoder_type = cfg.pipeline.categorical_encoder.type
            
            if encoder_type == "TargetEncoder":
                categorical_transformer = TargetEncoder()
            else:
                raise ValueError(f"Categorical encoder '{encoder_type}' is not available.")
            
            # Create a column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('categorical', categorical_transformer, OmegaConf.to_object(self.categorical_cols))
                ]
            )

            # Define the model
            model_type = cfg.pipeline.model.type
            model_params = cfg.pipeline.model.params

            if model_type == "GradientBoostingRegressor":
                model = GradientBoostingRegressor(**model_params)
            else:
                raise ValueError(f"Model type '{model_type}' is not available.")
            
            # Create the full pipeline
            steps = [
                ('preprocessor', preprocessor),
                ('model', model)
            ]

            logging.info("Pipeline created successfully.")
            return Pipeline(steps)
        
        except Exception as e:
            logging.error(f"Error in pipeline creation: {e}")
            raise

    def load_data(self):
        """
        Loads the training and testing datasets from the specified file paths.

        Returns:
        - bool: True if data loaded successfully, False otherwise.
        """
        try:
            # Get the current working directory
            data_dir = self.current_dir / self.data_path

            # Load train data
            train_file_path = data_dir / self.train_file
            logging.info(f"Loading training data from {train_file_path}")
            self.train_data = pd.read_csv(train_file_path)
            
            # Load test data
            test_file_path = data_dir / self.test_file
            logging.info(f"Loading testing data from {test_file_path}")
            self.test_data = pd.read_csv(test_file_path)

            logging.info("Training and testing data loaded successfully.")
            return True
        
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error while loading data: {e}")
            return False

    def prepare_data(self):
        """
        Prepares the data by transforming features using the pipeline.
        """
        try:
            # Determine train columns dynamically
            train_cols = [col for col in self.train_data.columns if col not in self.not_consider_column]
        
            self.X_train = self.train_data[train_cols]
            self.y_train = self.train_data[self.target_column]

            self.X_test = self.test_data[train_cols]
            self.y_test = self.test_data[self.target_column]

            logging.info("Data prepared successfully.")
        
        except KeyError as e:
            logging.error(f"Missing key column in the data: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during data preparation: {e}")
            raise

    def get_train_data(self):
        """
        Returns the training data.
        
        Returns:
        - tuple: X_train, y_train.
        """
        return self.X_train, self.y_train.values

    def get_test_data(self):
        """
        Returns the testing data.
        
        Returns:
        - tuple: X_test, y_test.
        """
        return self.X_test, self.y_test.values
    
