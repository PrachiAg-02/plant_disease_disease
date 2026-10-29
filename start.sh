#!/bin/bash

echo "=== CURRENT DIRECTORY & FILES ==="
pwd
ls -la

echo "=== Starting FastAPI Backend on port 8000 ==="
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &

echo "=== Starting Streamlit Frontend on port ${PORT:-10000} ==="
python -m streamlit run app.py --server.port ${PORT:-10000} --server.address 0.0.0.0 --server.headless true