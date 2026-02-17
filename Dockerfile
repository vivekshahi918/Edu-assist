# Use a lightweight Python base image
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .


EXPOSE 8501

# Command to run your app
# Streamlit:
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
