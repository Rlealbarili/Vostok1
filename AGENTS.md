# AGENTS.MD - Vostok-1 Knowledge Base

> **Status do Sistema:** 🟢 FASE 2 OPERACIONAL
> **Última Atualização:** 2024-12-16T15:58:00-03:00
> **Engenheiro Chefe:** Petrovich
> **Operador:** Vostok

## 1. Missão
Sistema de trading autônomo de baixa latência (Redis Streams), segregando ingestão, análise de sentimento (LLM) e execução quantitativa.

## 2. Arquitetura (Resumo)
- **Core:** Redis Streams (Barramento de Eventos)
- **Persistência:** TimescaleDB + PGVector
- **Linguagem:** Python 3.11 (Asyncio + ccxt.pro)
- **Módulos:** Ingestor → Sentiment → Quant → Execution

## 3. Estado Atual do Projeto
- [x] Definição de Arquitetura (DDP-VOSTOK-GENESIS)
- [x] Setup Docker Compose (Redis 7 + TimescaleDB PG16)
- [x] **Módulo Ingestor OPERACIONAL** ✅
- [ ] Módulo Quant (Processador)
- [ ] Módulo Sentiment (Qwen)
- [ ] Módulo Decision (Motor)
- [ ] Módulo Executor

## 4. Memória de Contexto

### Sessão 2024-12-16 - Fase 2 (Ingestor) ✅
**Ordem:** Capturar trades BTC/USDT Binance → Redis Streams.

**Implementação:**
- `src/ingestor/main.py`: ccxt.pro async + redis-py + backoff exponencial
- `Dockerfile.ingestor`: Multi-stage (python:3.11-slim)
- Logging estruturado JSON

**Validação (15:58):**
```
✔ vostok_ingestor → 1845+ trades processados
✔ stream:market:btc_usdt → Dados fluindo em tempo real
```

**Próximos Passos:**
1. ~~Implementar Módulo Ingestor~~ ✅
2. Implementar Módulo Quant (TA-Lib)
3. Configurar persistência no TimescaleDB

## 5. Estrutura Redis Streams

### `stream:market:btc_usdt`
| Campo | Tipo | Descrição |
|-------|------|-----------|
| price | string | Preço do trade |
| amount | string | Quantidade |
| side | string | 'buy' ou 'sell' |
| timestamp | string | Unix timestamp (ms) |
| symbol | string | Par (BTC/USDT) |
| trade_id | string | ID único da exchange |

**Exemplo:**
```
1765911471274-0
  price: 87724.15
  amount: 0.00115
  side: sell
  timestamp: 1765911471222
```

## 6. Árvore de Arquivos
```
VOSTOK1/
├── docker-compose.yml
├── Dockerfile.ingestor
├── src/
│   ├── ingestor/
│   │   ├── main.py          # WebSocket → Redis
│   │   └── requirements.txt
│   ├── quant/               # (Fase 3)
│   ├── sentiment/           # (Fase 3)
│   ├── decision/            # (Fase 4)
│   └── executor/            # (Fase 4)
└── scripts/init-db/
```

## 7. Diretrizes
- Nunca comitar chaves de API
- Type hints obrigatórios
- Logs JSON estruturados
- Priorizar `uvloop` (Linux)