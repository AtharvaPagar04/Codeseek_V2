# Fixture Infra Stack

Fixture Infra Stack defines local deployment services for an API, Postgres, Redis, and Qdrant.

## Usage

`docker compose up -d`

## Architecture

Docker Compose wires the API to Postgres, Redis, and Qdrant through service dependencies and environment variables.
