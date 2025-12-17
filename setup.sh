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

# ============================================================================
# CRIAR DIRETÓRIOS NECESSÁRIOS
# ============================================================================
echo "📁 Criando diretórios..."

DIRS=(
    "data/training"
    "models"
    "scripts/init-db"
)

for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
    echo "   ✓ $dir"
done

# ============================================================================
# INICIALIZAR ARQUIVOS DE DADOS
# ============================================================================
echo ""
echo "📄 Inicializando arquivos de dados..."

# Criar arquivo de dataset vazio se não existir
DATASET_FILE="data/training/dataset.jsonl"
if [ ! -f "$DATASET_FILE" ]; then
    touch "$DATASET_FILE"
    echo "   ✓ $DATASET_FILE (inicializado vazio)"
else
    LINES=$(wc -l < "$DATASET_FILE")
    echo "   ✓ $DATASET_FILE (existente: $LINES linhas)"
fi

# ============================================================================
# CONFIGURAR PERMISSÕES
# ============================================================================
echo ""
echo "🔐 Configurando permissões..."

chmod -R 777 data/ 2>/dev/null || true
chmod -R 777 models/ 2>/dev/null || true

echo "   ✓ data/ (777)"
echo "   ✓ models/ (777)"

# ============================================================================
# VERIFICAR AMBIENTE
# ============================================================================
echo ""
echo "🔍 Verificando ambiente..."

# Verificar .env
if [ ! -f .env ]; then
    echo "   ⚠️  .env não encontrado (copie de .env.example)"
else
    echo "   ✓ .env encontrado"
fi

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "   ❌ Docker não encontrado"
    exit 1
fi
echo "   ✓ Docker encontrado"

# ============================================================================
# BUILD (OPCIONAL)
# ============================================================================
if [ "$1" == "--build" ]; then
    echo ""
    echo "🔨 Construindo imagens Docker..."
    docker compose build --parallel
fi

# ============================================================================
# FINALIZAÇÃO
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    SETUP CONCLUÍDO!                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Estrutura criada:"
echo "  data/"
echo "  └── training/"
echo "      └── dataset.jsonl"
echo "  models/"
echo ""
echo "Comandos:"
echo "  docker compose up -d                              # Iniciar sistema"
echo "  docker compose run --rm --profile gui monitor     # Monitor TUI"
echo "  docker compose run --rm --profile batch trainer   # Treinar ML"
echo ""
