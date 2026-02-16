#!/bin/bash
set -e

echo "🚀 Starting LuminaLib..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from defaults..."
    cat > .env << 'EOF'
DATABASE_URL=postgresql+asyncpg://lumina:lumina@localhost:5432/luminalib
STORAGE_BACKEND=local
STORAGE_PATH=./storage
LLM_PROVIDER=mock
LLM_API_KEY=
REDIS_URL=redis://localhost:6379
SECRET_KEY=change-me-in-production-use-a-long-random-string
ACCESS_TOKEN_EXPIRE_MINUTES=60
EOF
fi

# Start services
echo "🐳 Starting Docker containers..."
docker-compose up --build -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check health
echo "🏥 Checking API health..."
curl -s http://localhost:5223/health | grep -q "healthy" && echo "✅ API is healthy!" || echo "⚠️  API might not be ready yet — retry in a few seconds"

echo ""
echo "✨ LuminaLib is running!"
echo ""
echo "📚 API Documentation: http://localhost:5223/docs"
echo "🔍 Health Check:      http://localhost:5223/health"
echo ""
echo "To stop: docker-compose down"
