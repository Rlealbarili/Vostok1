# ============================================================================
# VOSTOK-1 :: Setup Script (PowerShell - Windows)
# Prepara o ambiente de desenvolvimento e diretórios necessários
# ============================================================================

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           VOSTOK-1 SNIPER PROTOCOL - SETUP                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Criar diretórios
Write-Host "📁 Criando diretórios..." -ForegroundColor Yellow

$dirs = @("data\training", "models", "scripts\init-db")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   ✓ $dir" -ForegroundColor Green
    } else {
        Write-Host "   ○ $dir (já existe)" -ForegroundColor DarkGray
    }
}

# Inicializar arquivo de dataset
Write-Host ""
Write-Host "📄 Inicializando arquivos de dados..." -ForegroundColor Yellow

$datasetFile = "data\training\dataset.jsonl"
if (!(Test-Path $datasetFile)) {
    New-Item -ItemType File -Path $datasetFile -Force | Out-Null
    Write-Host "   ✓ $datasetFile (inicializado vazio)" -ForegroundColor Green
} else {
    $lines = (Get-Content $datasetFile | Measure-Object -Line).Lines
    Write-Host "   ○ $datasetFile (existente: $lines linhas)" -ForegroundColor DarkGray
}

# Verificar .env
Write-Host ""
Write-Host "🔍 Verificando ambiente..." -ForegroundColor Yellow

if (!(Test-Path ".env")) {
    Write-Host "   ⚠️  .env não encontrado" -ForegroundColor Yellow
} else {
    Write-Host "   ✓ .env encontrado" -ForegroundColor Green
}

# Verificar Docker
try {
    docker --version | Out-Null
    Write-Host "   ✓ Docker encontrado" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Docker não encontrado" -ForegroundColor Red
    exit 1
}

# Build (opcional)
if ($args -contains "--build") {
    Write-Host ""
    Write-Host "🔨 Construindo imagens Docker..." -ForegroundColor Yellow
    docker compose build --parallel
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    SETUP CONCLUÍDO!                         ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Comandos:" -ForegroundColor Cyan
Write-Host "  docker compose up -d                              # Iniciar sistema"
Write-Host "  docker compose run --rm --profile gui monitor     # Monitor TUI"
Write-Host "  docker compose run --rm --profile batch trainer   # Treinar ML"
Write-Host ""
