# Property Valuation Model Training and API Pipeline

---

## Project Overview

This project provides a pipeline for training and evaluating a machine learning model to predict **property valuations** based on various features. The pipeline is organized using **Hydra** for configuration management, **Scikit-learn** for model training, and **FastAPI** for serving predictions. The project includes:

1. **Data Loading and Preprocessing** using a configurable pipeline.
2. **Model Training and Evaluation**. This part can be used in pipeline and API.
3. **Saving the trained model** for use in serving predictions (e.g., via the API).

---

## Prerequisites

Before running the pipeline, ensure you have the following installed:

- **Python 3.9+**
- **Pip**
- **Conda**
- **Docker** (if using containers for experiments)

### Setting Up a Conda Environment (Recommended)

It is recommended to create a **Conda environment** for managing dependencies and isolating the project environment. This ensures that the required packages are installed in a clean environment, reducing compatibility issues. To create and activate a Conda environment, use the following commands:

```bash
conda create -n property-valuation python=3.9
conda activate property-valuation
```
  
Once the environment is activated, install the required Python libraries by running::

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
├── doc/
│   ├── Challenge.md                   # Challenge readme description
│   ├── Property-Friends-basic-model.ipynb  # Original model implementation using a jupyter notebook
|
├── pipeline/
│   ├── data_loader.py             # Data loading and preprocessing pipeline
│   ├── train.py                   # Model training and evaluation logic
│   ├── config.yaml                # Hydra configuration file 
|   ├── requirements.txt           # Python requirements file 
│   └── Dockerfile                 # Dockerfile for running training pipeline
│
├── data/                          # Data directory (Put the train/test data here when you have it)
│   ├── train.csv                  # Training data
│   ├── test.csv                   # Testing data
│
└── models/                        # Directory to save trained models
    └── trained_model.pkl          # Output model file
    
```
---
## Configuration (config.yaml)
The Hydra configuration file (config.yaml) handels all key settings for data paths, model parameters, and more. It enables easy configurability and adaptability of the pipeline.
Hydra was used due to its simplicity and ability to effectively manage complex comfigurations and nested parameters (This can be very helpful to continue the future development of the project).
For instance, additional configuration files can be created for various environments or experiments, and Hydra will automatically merge then with the base configuration. This allows for easy
switching between different configurations without modyfing the codebase.

In addition to data paths, **other parameters** such as the model type, model hyperparameters, and feature transformation methods (e.g., categorical encoders) are also configurable using this file.

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
  enabled: false  # Default option set to false. Set to true to enable database integration
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
- **Model and transformation configurations:** You can customize the model type, parameters, and categorical encoders directly in the config file.

---
## Running the Pipeline
1. **Training the Model:**
To train the model, simply run the ``train.py`` script. The script will:
    - Load the data from the specified location.
    - Preprocess the data using the specified categorical encoders and transformations.
    - Train the model with the configured parameters (e.g., GradientBoostingRegressor).
    - Evaluate and present the model performance results.
    - Save the trained model to the specified output path.

Command to Train the Model:
```bash
python pipeline/train.py
```
2. **Modifying the Configuration:**
It is possible to modify the ``config.yaml`` file to point to different folders or use different model parameters. For example, If you want to use a different directory for the data:
    - Provide the complete absolute path for both the data and model folder in the ``config.yaml`` file.

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
This project uses Hydra to manage experiment outputs. Each run creates a timestamped folder (saves outputs by date and time) under ``outputs``, formatted as ``outputs/YYYY-MM-DD/HH-MM-SS/``.

**Configuring Output Directory:**
You can customize the output location in ``config.yaml``:
```yaml
hydra:
  run:
    dir: ./custom_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}-${hydra.job.num}
```

For advanced options, refer to the Hydra documentation.

---
## Running the API without Docker
To run the FastAPI server without Docker, simply run the ``api.py`` script. 

```bash
python api.py
```
This will start the API server, and it will be available at http://localhost:8000 for serving predictions.
**Note:** The ``api.py`` script uses API keys for basic security. To view or change the key, check the ``api_key`` variable in the ``api.py`` script.

---
## Docker Usage
1. **Docker for Model Training:**
You can also run the training pipeline inside a Docker container. This is useful for ensuring consistent environments across machines. 

Building the Docker Image:
```bash
cd pipeline
docker build -t property-valuation-pipeline .
```
Before running the Docker container, ensure you have the necessary training and testing data in your local folder (``$(pwd)/data`` should contain ``train.csv`` and ``test.csv``).

Running the Docker Container:
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models property-valuation-pipeline
```
In this command:
- ``$(pwd)`` represents the current working directory on your machine.
- The data and models directories from your local machine are mounted inside the Docker container, allowing it to access and save files during the process.

This command will:
- Mount the data/ and models/ directories to the container. 
- Train the model inside the container.
- Save the trained model to the models/ directory inside the container.

2. **Docker for API:**
You can also serve the trained model using FastAPI inside a Docker container. In the main folder of the project run the following commands:

Building the FastAPI Docker Image:
```bash
docker build -t property-valuation-api .
```
Running the FastAPI Container:
```bash
docker run -d -p 8000:8000 property-valuation-api
```

This will expose the API on http://localhost:8000, where you can make predictions using the trained model.
Once the server is running, open a web browser and go to: http://127.0.0.1:8000/docs to test and access the API documentation.

It is possible to test prediction endpoint using cURL:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: your_secret_api_key' \
  -d '{
  "type": "departamento",
  "sector": "las condes",
  "net_usable_area": 140.0,
  "net_area": 165.0,
  "n_rooms": 4.0,
  "n_bathroom": 4.0,
  "latitude": -33.41135,
  "longitude": -70.56976999999998,
  "price": 14500
}'
```

On web browser can be test only using:
```bash
{
  "type": "departamento",
  "sector": "las condes",
  "net_usable_area": 140.0,
  "net_area": 165.0,
  "n_rooms": 4.0,
  "n_bathroom": 4.0,
  "latitude": -33.41135,
  "longitude": -70.56976999999998,
  "price": 14500
}
```

---
## Evaluation and Metrics
After training, the ``train.py`` script will automatically evaluate the model on the test dataset and log performance metrics like RMSE, MAPE, and MAE. These logs will be saved to ``train.log``.

--- 
## Future Database Integration
This project supports future integration with external databases like Amazon RDS or Azure SQL. To enable this:

1. Update the ``config.yaml`` file:
    - Set ``database.enabled: true``
    - Choose a provider (either aws or azure)
    - Fill in the connection details (e.g., ``host``, ``database``, ``user``, ``password``).
2. Customize the script to connect to the database using the ``connect_to_database()`` function defined in ``data_loader.py`` and call it from ``train.py``.

```python
if cfg.database.enabled:
    connect_to_database(cfg.database)
```
**Note:** Currently, the ``connect_to_database()`` function is a placeholder (not execute any operation). In the future, this integration could be used to log metrics, store results, or interact with external data sources.


---
## Future Improvements
1. **Automate Docker Commands:** Scripts can be created to automate building and running Docker images, reducing manual steps.
2. **Modularization:** Separate key functions such as data loading, preprocessing, training, and evaluation into dedicated modules for better reusability and maintainability.
3. **Simplified Commands:** Provide wrapper scripts or commands to simplify running common tasks (e.g., training, evaluation, serving the API).
4. **Environment Variables:** Sensitive information such as the API Key should be stored using environment variables rather than hardcoding them in the configuration files.
5. **Improved Configuration Management:** Use Hydra to define different configuration files for different environments (e.g., dev, test, production).
6. **Target Variable Used in Training:** Currently, the target variable is included in the model training. This was not changed to respect the original model received, but it should be corrected for future models to ensure proper training and evaluation.
7. **Integration with MLflow or other similar tool for manage ML workflow:** For experiment tracking and model versioning, integrating with MLflow would allow better management of experiments, logging, and model lifecycle tracking.


## Troubleshooting
- **File Not Found Errors:** Ensure that the ``data_path``, ``train_file``, and ``test_file`` are correctly set in the ``config.yaml`` file.
- **Missing Dependencies:** If you encounter missing libraries, ensure that all requirements are installed via ``pip install -r requirements.txt``.
