#!/bin/bash

# 1. Start FastAPI in the background on internal port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 &

# 2. Start Streamlit on Render's dynamic PORT (user-facing frontend)
streamlit run app.py --server.port ${PORT:-10000} --server.address 0.0.0.0