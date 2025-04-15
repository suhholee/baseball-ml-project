FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Environment variables
ENV PORT=8080

# Command to run the application
CMD python -m functions_framework --target=scrape_baseball_data --port=${PORT}
