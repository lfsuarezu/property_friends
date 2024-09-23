# Property-Friends Real State case
> This is an hypothetical scenario and is not related with an actual project.

## Context
Your team has been assigned the task of developing a model to estimate property valuations for an important real estate client in Chile. The idea of the client is deliver quickly a model (even if not the best one) into production so the team can develop the required infraestructure for supporting future new models/projects.

The client requires a streamlined process for conducting property appraisals, and they seek to leverage machine learning techniques to achieve quick and reliable estimates. Your team was responsible for building a model that can predict the cost of residential properties in Chile based on property features.

After some experimentations the Client is happy with the current results and wants to bring the model into a deployable solution.

## Task Required

Your task is to productivize the existing Jupyter Notebook, ensuring that coding best practices are followed throughout the process. You will be responsible for creating a robust pipeline that automates the model training, evaluation, and deployment process. Additionally, you need to develop an API that can receive property information and provide accurate valuation predictions.

It is crucial to document any assumptions made during the development of the model and highlight potential areas of improvement. To facilitate this, you are required to provide a comprehensive README file explaining the project's structure, dependencies, and instructions for running the pipeline and API.

### Material Provided:
1. **Property-Friends-basic-model.ipynb:** Notebook with the actual code that trains the model.
2. **train.csv, test.csv**: Datasets used for training and evaluating the model.

### Deliverables:
1. A well-structured pipeline that trains the property valuation model using the provided dataset, ensuring reproducibility and scalability.
2. An API that can receive property information and generate accurate valuation predictions.
3. A README file documenting the project, including instructions for running the pipeline and API, dependencies, and any assumptions or suggestions for improvement.

Right now the model uses two files: `train.csv`, `test.csv` for training. In the future the client wants to connects the pipeline directly into their databases, they asked for us to create the abstractions for acomplish this on the future.

Some technicals requirements required from the Software Engineers and MLEngineers:
1. The pipeline and API should run using Docker.
2. API calls/predictions should generate logs using a logger for future model monitoring.
3. API should be documented, its recomendable to use a framework as FastAPI or similar.
4. The client asked for basic security system for the API (API-Keys or similar).

> The deliverable should be a Github repository containing all the requested. The repository shouldn't contain the data due is property of the client.

**Note**: As you develop the pipeline and API, adhere to coding best practices, including modularization, documentation, and error handling. Consider potential edge cases and handle them appropriately to ensure reliable performance. The client values clean, maintainable code that can be easily scaled and extended in the future.