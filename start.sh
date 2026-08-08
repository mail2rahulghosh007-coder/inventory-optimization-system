#!/bin/bash
# IMPORTANT: api.py uses relative paths ('models', not an absolute path
# resolved from the file's own location) -- so uvicorn MUST be started
# with the working directory at /app (the project root), NOT inside app/.
# Using 'uvicorn app.api:app' (dotted module path) instead of 'cd app &&
# uvicorn api:app' keeps the working directory correct at /app, so
# api.py's relative 'models' path correctly resolves to /app/models.

set -e

echo "Starting FastAPI backend..."
uvicorn app.api:app --host 127.0.0.1 --port 8000 &

sleep 5

echo "Starting Streamlit frontend..."
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0