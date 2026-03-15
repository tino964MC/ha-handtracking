#!/usr/bin/with-contenv bashio

bashio::log.info "Starting Hand Control..."
exec python3 /app/hand_tracker.py