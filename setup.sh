#!/bin/bash

# ============================================================================
# VOSTOK-1 SETUP SCRIPT
# Inicializa a estrutura de pastas e volumes para o projeto
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  VOSTOK-1 :: Iniciando Setup da Fase 1"
echo "  Engenheiro Chefe: Petrovich"
echo "=============================================="
echo ""

# Diretório base do projeto
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Lista de diretórios a serem criados
DIRS=(
    # Volumes Docker
    "data/redis"
    "data/timescale"
    
    # Scripts de inicialização do banco
    "scripts/init-db"
    
    # Módulos do sistema (Fase 2+)
    "src/ingestor"
    "src/sentiment"
    "src/quant"
    "src/decision"
    "src/executor"
    "src/common"
    
    # Configurações
    "config"
    
    # Logs estruturados
    "logs"
    
    # Testes
    "tests"
)

echo "[1/3] Criando estrutura de diretórios..."
for dir in "${DIRS[@]}"; do
    mkdir -p "$PROJECT_ROOT/$dir"
    echo "  ✓ $dir"
done

echo ""
echo "[2/3] Criando arquivos de inicialização..."

# Gitkeep para diretórios vazios
touch "$PROJECT_ROOT/logs/.gitkeep"
touch "$PROJECT_ROOT/data/redis/.gitkeep"
touch "$PROJECT_ROOT/data/timescale/.gitkeep"

# Script de inicialização do banco (PGVector + Hypertables)
cat > "$PROJECT_ROOT/scripts/init-db/01-init-extensions.sql" << 'EOF'
-- ============================================================================
-- VOSTOK-1 DATABASE INITIALIZATION
-- Extensões necessárias: TimescaleDB + PGVector
-- ============================================================================

-- Habilitar extensões
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;

-- Confirmar instalação
SELECT extname, extversion FROM pg_extension 
WHERE extname IN ('timescaledb', 'vector');
EOF

echo "  ✓ scripts/init-db/01-init-extensions.sql"

# Arquivo .env de exemplo
cat > "$PROJECT_ROOT/.env.example" << 'EOF'
# ============================================================================
# VOSTOK-1 ENVIRONMENT VARIABLES
# Copie para .env e configure os valores reais
# ============================================================================

# PostgreSQL/TimescaleDB
POSTGRES_PASSWORD=vostok_secure_2024

# Redis (opcional, usa default se não especificado)
# REDIS_PASSWORD=

# Exchange APIs (Fase 4)
# BINANCE_API_KEY=
# BINANCE_SECRET_KEY=
EOF

echo "  ✓ .env.example"

# Criar __init__.py nos módulos src
for module in ingestor sentiment quant decision executor common; do
    touch "$PROJECT_ROOT/src/$module/__init__.py"
done
echo "  ✓ src/*/__init__.py (módulos Python)"

echo ""
echo "[3/3] Verificando estrutura final..."
echo ""

# Exibir árvore (se disponível)
if command -v tree &> /dev/null; then
    tree -L 2 "$PROJECT_ROOT" --dirsfirst -I '.git|__pycache__|*.pyc'
else
    echo "Estrutura criada:"
    find "$PROJECT_ROOT" -type d -not -path '*/\.*' | head -30
fi

echo ""
echo "=============================================="
echo "  ✓ Setup concluído com sucesso!"
echo "  Próximo passo: docker compose up -d"
echo "=============================================="
