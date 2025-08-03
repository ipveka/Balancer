#!/bin/bash

# Balancer Platform Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="balancer"
CONTAINER_NAME="balancer-app"
PORT=8000

echo -e "${GREEN}🚀 Balancer Platform Deployment Script${NC}"
echo "========================================"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_warning "Docker Compose not found. Trying docker compose..."
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    DOCKER_COMPOSE_CMD="docker compose"
else
    DOCKER_COMPOSE_CMD="docker-compose"
fi

# Parse command line arguments
ENVIRONMENT="production"
BUILD_ONLY=false
STOP_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --stop)
            STOP_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env ENV        Set environment (development|production) [default: production]"
            echo "  --build-only     Only build the image, don't run"
            echo "  --stop           Stop and remove containers"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Environment: $ENVIRONMENT"

# Stop containers if requested
if [ "$STOP_ONLY" = true ]; then
    print_status "Stopping containers..."
    $DOCKER_COMPOSE_CMD down
    print_status "Containers stopped successfully"
    exit 0
fi

# Create logs directory
mkdir -p logs

# Copy environment file
if [ "$ENVIRONMENT" = "production" ]; then
    if [ -f ".env.production" ]; then
        cp .env.production .env
        print_status "Using production environment configuration"
    else
        print_warning "Production environment file not found, using defaults"
    fi
elif [ "$ENVIRONMENT" = "development" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status "Using development environment configuration"
    else
        print_warning "Development environment file not found, using defaults"
    fi
fi

# Build the Docker image
print_status "Building Docker image..."
docker build -t $IMAGE_NAME:latest .

if [ $? -eq 0 ]; then
    print_status "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Exit if build-only flag is set
if [ "$BUILD_ONLY" = true ]; then
    print_status "Build completed. Use 'docker run -p $PORT:$PORT $IMAGE_NAME:latest' to run the container."
    exit 0
fi

# Stop existing containers
print_status "Stopping existing containers..."
$DOCKER_COMPOSE_CMD down

# Start the application
print_status "Starting Balancer platform..."
$DOCKER_COMPOSE_CMD up -d

# Wait for the application to start
print_status "Waiting for application to start..."
sleep 10

# Health check
print_status "Performing health check..."
for i in {1..30}; do
    if curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
        print_status "✅ Application is healthy and running!"
        break
    fi
    
    if [ $i -eq 30 ]; then
        print_error "❌ Health check failed after 30 attempts"
        print_status "Checking container logs..."
        $DOCKER_COMPOSE_CMD logs balancer
        exit 1
    fi
    
    echo -n "."
    sleep 2
done

echo ""
print_status "🎉 Deployment completed successfully!"
echo ""
echo "Application URLs:"
echo "  - API: http://localhost:$PORT"
echo "  - Health Check: http://localhost:$PORT/health"
echo "  - API Documentation: http://localhost:$PORT/docs"
echo "  - API Status: http://localhost:$PORT/api/v1/status"
echo ""
echo "Useful commands:"
echo "  - View logs: $DOCKER_COMPOSE_CMD logs -f balancer"
echo "  - Stop application: $DOCKER_COMPOSE_CMD down"
echo "  - Restart application: $DOCKER_COMPOSE_CMD restart balancer"
echo ""
print_status "Deployment completed! 🚀"