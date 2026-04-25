#!/usr/bin/env bash
set -euo pipefail

# Stops the local Weaviate instance started by setup/start_weaviate.sh.

docker compose -f setup/docker-compose.weaviate.yml down
