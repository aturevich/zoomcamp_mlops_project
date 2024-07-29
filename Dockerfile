# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install pipenv and dependencies
RUN pip install pipenv && \
    pipenv install --deploy --system && \
    pip uninstall pipenv -y

# Copy the rest of the application
COPY . .

# Expose port for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]