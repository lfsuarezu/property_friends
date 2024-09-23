# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Copy the API code into the container
COPY api.py .
COPY models\ ./models/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run FastAPI when the container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]