#!/bin/bash
# ============================================================================
# VOSTOK-1 :: Setup Script
# Prepara o ambiente de desenvolvimento e diretórios necessários
# ============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           VOSTOK-1 SNIPER PROTOCOL - SETUP                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Criar diretórios necessários
echo "📁 Criando diretórios..."
mkdir -p data
mkdir -p models
mkdir -p scripts/init-db

# Permissões (para evitar problemas com Docker em Linux)
echo "🔐 Configurando permissões..."
chmod -R 755 data models 2>/dev/null || true

# Verificar .env
if [ ! -f .env ]; then
    echo "⚠️  Arquivo .env não encontrado!"
    echo "   Copie .env.example para .env e configure suas chaves."
else
    echo "✅ Arquivo .env encontrado"
fi

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado. Por favor, instale o Docker."
    exit 1
fi

echo "✅ Docker encontrado"

# Build das imagens
echo ""
echo "🔨 Construindo imagens Docker..."
docker compose build --parallel

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    SETUP CONCLUÍDO!                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Próximos passos:"
echo "  1. Iniciar sistema: docker compose up -d"
echo "  2. Ver monitor:     docker compose run --rm --profile gui monitor"
echo "  3. Treinar modelo:  docker compose run --rm --profile batch trainer"
echo ""
