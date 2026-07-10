# Fixture Backend API

Fixture Backend API exposes FastAPI endpoints for repository search and stores vectors in Qdrant.

## Setup

`pip install -r requirements.txt`

## Usage

`uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Architecture

FastAPI receives requests, Qdrant stores embeddings, and Uvicorn serves the app process.
