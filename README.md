# Property Valuation Model Training and API Pipeline

---

## Project Overview

This project provides a pipeline for training and evaluating a machine learning model to predict **property valuations** based on various features. The pipeline is organized using **Hydra** for configuration management, **Scikit-learn** for model training, and **FastAPI** for serving predictions. The project includes:

1. **Data Loading and Preprocessing** using a configurable pipeline.
2. **Model Training and Evaluation**.
3. **Saving the trained model** for future use in serving predictions.

---

## Prerequisites

Before running the pipeline, ensure you have the following installed:

- **Python 3.9+**
- **Pip**
- **Docker** (if using containers for experiments)
  
Install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

---
## Directory Structure
The project structure is organized as follows:
```
property-valuation/
│
├── api.py                     # FastAPI code for serving predictions
├── Dockerfile                 # Dockerfile for FastAPI
│
├── pipeline/
│   ├── data_loader.py             # Data loading and preprocessing pipeline
│   ├── train.py                   # Model training and evaluation logic
│   ├── config.yaml                # Hydra configuration file 
|   ├── requirements.txt           # Python requirements file 
│   └── Dockerfile                 # Dockerfile for running training pipeline
│
├── data/                          # Data directory (Put the train/test data when you have it)
│   ├── train.csv                  # Training data
│   ├── test.csv                   # Testing data
│
└── models/                        # Directory to save trained models
    └── trained_model.pkl          # Output model file
```
---
## Configuration (config.yaml)
The Hydra configuration file (config.yaml) manages all key settings for data paths, model parameters, and more. It makes the pipeline easily configurable and adaptable.

Sample config.yaml:
``` yaml
hydra:
    output_subdir: Null

# Data loader settings configuration
data_loader:
    data_path: 'data'              # Directory for data
    train_file: 'train.csv'        # Training file name
    test_file: 'test.csv'          # Testing file name
    target_column: "price"         # Target column for property price
    not_consider_column:           # Columns to exclude
        - "id"
    categorical_cols:              # Categorical columns
        - "type"
        - "sector"

 
# Pipeline parameters configuration - Preprocessing and model     
pipeline:
    categorical_encoder:
        type: "TargetEncoder"  # Ensure you have the appropriate import for TargetEncoder
    model:
        type: "GradientBoostingRegressor"
        params:
            learning_rate: 0.01
            n_estimators: 300
            max_depth: 5
            loss: "absolute_error"
    model_path: 'models'

# Example for database integration settings (for future use)
database:
  enabled: false  # Set to true to enable database integration
  provider: aws   # Options: aws, azure
  aws:
    host: "your-rds-instance.amazonaws.com"
    port: 3306
    database: "your_database_name"
    user: "your_username"
    password: "your_password"
  azure:
    host: "your-azure-sql-server.database.windows.net"
    port: 1433
    database: "your_database_name"
    user: "your_username"
    password: "your_password"    
```
Configurable Parameters:
- ``data_loader.data_path``: Path to the directory containing the data (train.csv, test.csv).
- ``data_loader.train_file / test_file``: File names for training and testing datasets.
- ``pipeline.model_path``: Directory where the trained model will be saved.
You can modify the paths in ``config.yaml`` to point to different datasets or model output locations.

---
## Running the Pipeline
1. **Training the Model:**
To train the model, simply run the ``train.py`` script. The script will:
    - Load the data from the specified location.
    - Preprocess the data using the specified categorical encoders and transformations.
    - Train the model with the configured parameters (e.g., GradientBoostingRegressor).
    - Save the trained model to the specified output path.

Command to Train the Model:
```bash
python pipeline/train.py
```
2. **Modifying the Configuration:**
It is possible to modify the ``config.yaml`` file to point to different folders or use different model parameters. For example, If you want to use a different directory for the data, you have two options:
    - Respect the existing folder hierarchy as shown in the project structure. For example, place your ``train.csv`` and ``test.csv`` files in the ``data/`` folder and the trained model will automatically be saved to the ``models/`` folder.
    - Provide the complete absolute path for both the data and model folder in the ``config.yaml`` file or when running ``train.py``.

    For example:
```yaml
    data_loader:
        data_path: '/complete/path/to/data'  # Absolute path to data directory
    pipeline:
        model_path: '/complete/path/to/models'  # Absolute path to save models
```
3. **Running the Pipeline with Custom Data and Model Paths:**
It is possible to override the default configuration in ``config.yaml`` by passing the paths directly from the command line when running ``train.py``. For example, to specify new paths for the data folder and the model folder, use the following command:

```bash
python pipeline/train.py data_loader.data_path="/complete/path/to/data" pipeline.model_path="/complete/path/to/models"
```
This command will load the data from ``/complete/path/to/data``. and save the trained model to ``/complete/path/to/models``.

---
### Hydra Experiment Outputs
This project uses Hydra to manage experiment outputs. Each run creates a timestamped folder under ``outputs``, formatted as ``outputs/YYYY-MM-DD/HH-MM-SS/``.

Configuring Output Directory
You can customize the output location in ``config.yaml``:
```yaml
hydra:
  run:
    dir: ./custom_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}-${hydra.job.num}
```
This saves outputs by date and time.

For advanced options, refer to the Hydra documentation.

---
## Docker Usage
1. Docker for Model Training
You can also run the training pipeline inside a Docker container. This is useful for ensuring consistent environments across machines.

Building the Docker Image:
```bash
cd pipeline
docker build -t property-valuation-pipeline .
```
Running the Docker Container:
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models property-valuation-pipeline
```

This command will:
- Mount the data/ and models/ directories to the container.
- Train the model inside the container.
- Save the trained model to the models/ directory.

2. Docker for API
You can also serve the trained model using FastAPI inside a Docker container.

Building the FastAPI Docker Image:
```bash
cd api
docker build -t property-valuation-api .
```
Running the FastAPI Container:
```bash
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models property-valuation-api
```
This will expose the API on http://localhost:8000, where you can make predictions using the trained model.

---
## Evaluation and Metrics
After training, the train.py script will automatically evaluate the model on the test dataset and log performance metrics like RMSE, MAPE, and MAE. These logs will be saved to train.log.

--- 
## Future Database Integration
This project supports future integration with external databases like Amazon RDS or Azure SQL. To enable this:

1. Update the ``config.yaml`` file:
    - Set ``database.enabled: true``
    - Choose a provider (either aws or azure)
    - Fill in the connection details (e.g., ``host``, ``database``, ``user``, ``password``).
2. To customize the script to connect to the database, you can use the function defined in ``data_loader.py`` and call in main in ``train.py``, using the configured credentials.
```python
if cfg.database.enabled:
    connect_to_database(cfg.database)
```
This integration can be used to log metrics, store results, or interact with external data sources.

---
## Troubleshooting
File Not Found Errors: Ensure that the data_path, train_file, and test_file are correctly set in the ``config.yaml`` file.
Missing Dependencies: If you encounter missing libraries, ensure that all requirements are installed via ``pip install -r requirements.txt``.
