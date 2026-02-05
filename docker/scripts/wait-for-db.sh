#!/bin/sh
# Wait for PostgreSQL to be ready
# Usage: wait-for-db.sh <host> <port> [timeout]

set -e

HOST="${1:-postgres}"
PORT="${2:-5432}"
TIMEOUT="${3:-30}"

echo "Waiting for PostgreSQL at $HOST:$PORT..."

start_time=$(date +%s)
while ! nc -z "$HOST" "$PORT" 2>/dev/null; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "Timeout waiting for PostgreSQL after ${TIMEOUT}s"
        exit 1
    fi

    echo "PostgreSQL not ready yet... (${elapsed}s elapsed)"
    sleep 1
done

echo "PostgreSQL is ready!"
