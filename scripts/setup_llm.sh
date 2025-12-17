#!/bin/bash
# ============================================================================
# VOSTOK-1 :: LLM Setup Script
# Configura o Ollama com modelo Qwen 2.5 para análise de sentimento
# ============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           VOSTOK-1 :: LLM ENGINE SETUP                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

CONTAINER_NAME="vostok_llm"
MODEL_NAME="qwen2.5:7b-instruct"

# ============================================================================
# STEP 1: Verificar container
# ============================================================================
echo "🔍 Verificando container LLM Engine..."

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ Container '${CONTAINER_NAME}' não está rodando!"
    echo ""
    echo "Execute primeiro:"
    echo "  docker compose up -d llm_engine"
    exit 1
fi

echo "✅ Container '${CONTAINER_NAME}' está rodando"
echo ""

# ============================================================================
# STEP 2: Pull do modelo Qwen
# ============================================================================
echo "🧠 Baixando modelo ${MODEL_NAME}..."
echo "   (Isso pode levar alguns minutos na primeira vez)"
echo ""

docker exec -it ${CONTAINER_NAME} ollama pull ${MODEL_NAME}

echo ""
echo "✅ Modelo ${MODEL_NAME} instalado!"
echo ""

# ============================================================================
# STEP 3: Teste do modelo
# ============================================================================
echo "🧪 Testando modelo com pergunta simples..."
echo ""

TEST_RESPONSE=$(docker exec ${CONTAINER_NAME} ollama run ${MODEL_NAME} "Hello, are you ready? Reply in one sentence." 2>/dev/null)

if [ -n "$TEST_RESPONSE" ]; then
    echo "📝 Resposta do modelo:"
    echo "   \"${TEST_RESPONSE}\""
    echo ""
    echo "✅ Modelo funcionando corretamente!"
else
    echo "⚠️  Modelo não respondeu. Verifique os logs:"
    echo "   docker logs ${CONTAINER_NAME}"
fi

# ============================================================================
# STEP 4: Listar modelos instalados
# ============================================================================
echo ""
echo "📋 Modelos instalados:"
docker exec ${CONTAINER_NAME} ollama list

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    SETUP CONCLUÍDO!                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "API disponível em: http://localhost:11434"
echo ""
echo "Testar via API:"
echo '  curl http://localhost:11434/api/generate -d '"'"'{"model":"qwen2.5:7b-instruct","prompt":"BTC sentiment?"}'\'
echo ""
