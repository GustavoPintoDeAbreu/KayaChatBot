#!/bin/bash

# Kaya Chatbot Setup Script
# This script sets up the web application stack

set -e

echo "🚀 Setting up Kaya Chatbot Web App"
echo "=================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "   Please copy .env.example to .env and fill in your values:"
    echo "   cp .env.example .env"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

echo "📦 Building and starting services..."

# Build and start all services except dev
docker-compose up -d --build ollama api frontend cloudflared

echo "⏳ Waiting for Ollama to be ready..."
sleep 10

# Check if Ollama is running
if ! docker-compose exec -T ollama ollama list > /dev/null 2>&1; then
    echo "❌ Ollama is not responding. Please check the logs:"
    echo "   docker-compose logs ollama"
    exit 1
fi

echo "🤖 Creating Ollama model..."

# Create the model in Ollama
docker-compose exec -T ollama ollama create kaya-chatbot -f /root/.ollama/models/../Modelfile

echo "✅ Model created successfully!"

echo "🌐 Checking services status..."

# Check all services
docker-compose ps

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Your Kaya Chatbot is now running at:"
echo "• Frontend: http://localhost:3000"
echo "• API: http://localhost:8000"
echo "• Ollama: http://localhost:11434"
echo ""
echo "If you set up Cloudflare Tunnel, it should also be available at your custom domain."
echo ""
echo "To view logs: docker-compose logs -f [service-name]"
echo "To stop: docker-compose down"