# AGENTS.MD - Vostok-1 Knowledge Base

> **Status do Sistema:** 🟢 FASE 1 COMPLETA
> **Última Atualização:** 2024-12-16T15:30:00-03:00
> **Engenheiro Chefe:** Petrovich
> **Operador:** Vostok

## 1. Missão
Construir um sistema de trading autônomo de baixa latência baseado em eventos (Redis Streams), segregando ingestão de dados, análise de sentimento (LLM Local) e execução quantitativa.

## 2. Arquitetura (Resumo)
- **Core:** Redis Streams (Barramento de Eventos).
- **Persistência:** TimescaleDB (Séries Temporais) + PGVector.
- **Linguagem:** Python 3.11 (Asyncio).
- **Módulos:**
  1. `Ingestor` (WebSockets -> Redis)
  2. `Sentiment` (News API -> Qwen -> Redis)
  3. `Quant` (Redis -> TA-Lib -> Redis)
  4. `Execution` (Redis -> Exchange API)

## 3. Estado Atual do Projeto
- [x] Definição de Arquitetura (DDP-VOSTOK-GENESIS).
- [x] Configuração do Repositório (.gitignore).
- [x] Setup do Docker Compose (Redis 7 + TimescaleDB PG16).
- [ ] Implementação do Módulo Ingestor.

## 4. Memória de Contexto (Context Compression)

### Sessão 2024-12-16 - Fase 1 Concluída ✅
**Ordem:** Engenheiro Chefe Petrovich - Setup inicial da infraestrutura.

**Ações Realizadas:**
1. Criada estrutura de pastas modular: `src/`, `data/`, `scripts/`, `config/`, `logs/`, `tests/`
2. `docker-compose.yml` configurado:
   - **Redis 7 Alpine**: AOF, maxmemory 512MB, porta 6379
   - **TimescaleDB PG16-HA**: PGVector, shm_size 256MB, porta **5433**
   - **Volumes Docker nomeados** (evita problemas de permissão Windows)
3. Scripts e configurações: `setup.sh`, `.gitignore`, `.env.example`, `01-init-extensions.sql`

**Validação Final (2024-12-16 15:49):**
```
✔ vostok_redis     → PONG (healthy)
✔ vostok_timescale → timescaledb 2.24.0, vector 0.8.1 (healthy)
```

**Próximos Passos:**
1. ~~Executar `docker compose up -d`~~ ✅
2. ~~Validar conexão Redis e TimescaleDB~~ ✅
3. Iniciar implementação do Módulo `Ingestor`

## 5. Árvore de Arquivos
```
VOSTOK1/
├── docker-compose.yml      # Infraestrutura containerizada
├── setup.sh                # Script de inicialização (bash)
├── .gitignore
├── .env.example
├── AGENTS.md               # Este arquivo
├── DDP-VOSTOK-GENESIS.md   # Documento de Design
├── config/
├── data/
│   ├── redis/              # Volume Redis (AOF)
│   └── timescale/          # Volume PostgreSQL
├── logs/
├── scripts/
│   └── init-db/
│       └── 01-init-extensions.sql
├── src/
│   ├── common/             # Utilitários compartilhados
│   ├── decision/           # Motor de Decisão
│   ├── executor/           # Executor de Ordens
│   ├── ingestor/           # Ingestão de Mercado
│   ├── quant/              # Processador Quantitativo
│   └── sentiment/          # Análise de Sentimento AI
└── tests/
```

## 6. Diretrizes de Desenvolvimento
- Nunca comitar chaves de API.
- Manter `requirements.txt` mínimo.
- Priorizar `uvloop` para performance.
- Type hints obrigatórios.
- Logs em JSON estruturado.