#!/bin/bash
# Deployment script for Addis Ride Demand Forecast App

set -e

echo "======================================"
echo "Addis Ride Forecast - Deployment"
echo "======================================"

# Stop and remove existing containers
echo "→ Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Remove old images
echo "→ Cleaning up old images..."
docker system prune -f

# Build new image
echo "→ Building Docker image..."
docker-compose build --no-cache

# Start the application
echo "→ Starting application..."
docker-compose up -d

# Wait for health check
echo "→ Waiting for application to be ready..."
sleep 10

# Check if running
if docker ps | grep -q addis-ride-forecast; then
    echo "✓ Application is running!"
    echo ""
    echo "Access the app at: http://3.126.82.53/"
    echo ""
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
else
    echo "✗ Application failed to start"
    echo "Check logs: docker-compose logs"
    exit 1
fi
