# ============================================================================
# VOSTOK-1 :: LLM Setup Script (PowerShell - Windows)
# Configura o Ollama com modelo Qwen 2.5 para análise de sentimento
# ============================================================================

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           VOSTOK-1 :: LLM ENGINE SETUP                      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$CONTAINER_NAME = "vostok_llm"
$MODEL_NAME = "qwen2.5:7b-instruct"

# ============================================================================
# STEP 1: Verificar container
# ============================================================================
Write-Host "🔍 Verificando container LLM Engine..." -ForegroundColor Yellow

$runningContainers = docker ps --format '{{.Names}}'
if ($runningContainers -notcontains $CONTAINER_NAME) {
    Write-Host "❌ Container '$CONTAINER_NAME' não está rodando!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Execute primeiro:" -ForegroundColor Yellow
    Write-Host "  docker compose up -d llm_engine"
    exit 1
}

Write-Host "✅ Container '$CONTAINER_NAME' está rodando" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 2: Pull do modelo Qwen
# ============================================================================
Write-Host "🧠 Baixando modelo $MODEL_NAME..." -ForegroundColor Yellow
Write-Host "   (Isso pode levar alguns minutos na primeira vez)" -ForegroundColor DarkGray
Write-Host ""

docker exec -it $CONTAINER_NAME ollama pull $MODEL_NAME

Write-Host ""
Write-Host "✅ Modelo $MODEL_NAME instalado!" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 3: Teste do modelo
# ============================================================================
Write-Host "🧪 Testando modelo com pergunta simples..." -ForegroundColor Yellow
Write-Host ""

$testResponse = docker exec $CONTAINER_NAME ollama run $MODEL_NAME "Hello, are you ready? Reply in one sentence." 2>$null

if ($testResponse) {
    Write-Host "📝 Resposta do modelo:" -ForegroundColor Cyan
    Write-Host "   `"$testResponse`"" -ForegroundColor White
    Write-Host ""
    Write-Host "✅ Modelo funcionando corretamente!" -ForegroundColor Green
}
else {
    Write-Host "⚠️  Modelo não respondeu. Verifique os logs:" -ForegroundColor Yellow
    Write-Host "   docker logs $CONTAINER_NAME"
}

# ============================================================================
# STEP 4: Listar modelos instalados
# ============================================================================
Write-Host ""
Write-Host "📋 Modelos instalados:" -ForegroundColor Cyan
docker exec $CONTAINER_NAME ollama list

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    SETUP CONCLUÍDO!                         ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "API disponível em: http://localhost:11434" -ForegroundColor Cyan
Write-Host ""
Write-Host "Testar via API:" -ForegroundColor Yellow
Write-Host '  Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body ''{"model":"qwen2.5:7b-instruct","prompt":"BTC sentiment?"}'''
Write-Host ""
