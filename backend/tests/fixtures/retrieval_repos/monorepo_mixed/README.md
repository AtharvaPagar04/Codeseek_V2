# Fixture Mixed Monorepo

Fixture Mixed Monorepo contains a React web app, a FastAPI service, and Docker Compose infrastructure.

## Setup

`npm install --prefix apps/web`

`pip install -r services/api/requirements.txt`

## Usage

`docker compose up -d`

## Architecture

The Vite React frontend calls the FastAPI backend, while Docker Compose starts web, api, and qdrant services together.
