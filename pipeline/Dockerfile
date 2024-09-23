# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create folders for data, models, and pipeline
RUN mkdir -p /app/data /app/models /app/pipeline

# Copy project files into the appropriate folders
COPY config.yaml /app/pipeline/
COPY data_loader.py /app/pipeline/
COPY train.py /app/pipeline/


# Set the default command to run the training script when the container starts
CMD ["python", "/app/pipeline/train.py"]
