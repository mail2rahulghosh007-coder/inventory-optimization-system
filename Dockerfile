FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (FastAPI + Streamlit)
COPY app/ ./app/

# Trained model artifacts
COPY models/model_perishable.pkl ./models/model_perishable.pkl
COPY models/model_nonperishable.pkl ./models/model_nonperishable.pkl
COPY models/feature_list.pkl ./models/feature_list.pkl

# Data files needed at inference time (NOT the huge raw Kaggle files --
# only what app/api.py and app/streamlit_app.py actually read at runtime.
# Adjust this list to match what your app code actually loads.)
COPY data/retail_features.csv ./data/retail_features.csv

COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8501

CMD ["./start.sh"]