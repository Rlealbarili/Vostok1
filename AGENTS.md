# AGENTS.MD - Vostok-1 Knowledge Base

> **Status do Sistema:** 🟢 FASE 3A OPERACIONAL
> **Última Atualização:** 2024-12-16T16:15:00-03:00
> **Engenheiro Chefe:** Petrovich
> **Operador:** Vostok

## 1. Missão
Sistema de trading autônomo de baixa latência (Redis Streams), segregando ingestão, análise de sentimento (LLM) e execução quantitativa.

## 2. Arquitetura
```
Binance WS → Ingestor → stream:market:btc_usdt → Quant → stream:signals:tech
```

## 3. Estado Atual do Projeto
- [x] Definição de Arquitetura (DDP-VOSTOK-GENESIS)
- [x] Setup Docker Compose (Redis 7 + TimescaleDB PG16)
- [x] **Módulo Ingestor OPERACIONAL** ✅
- [x] **Módulo Quant OPERACIONAL** ✅
- [ ] Módulo Sentiment (Qwen)
- [ ] Módulo Decision (Motor)
- [ ] Módulo Executor

## 4. Memória de Contexto

### Sessão 2024-12-16 - Fase 3A (Quant) ✅
**Ordem:** Agregar ticks em OHLCV, calcular RSI/MACD/BB, publicar sinais.

**Implementação:**
- `src/quant/main.py`: Consumer Group + CandleManager + TA-Lib
- `Dockerfile.quant`: Multi-stage com TA-Lib C compilado
- Indicadores: RSI(14), MACD(12,26,9), Bollinger(20,2)

**Validação (16:15):**
```
✔ vostok_quant      → Up, Healthy
✔ Consumer Group    → quant_group (86k+ ticks processados)
✔ Stream signals    → Aguardando 26 velas para MACD
```

## 5. Estrutura Redis Streams

### `stream:market:btc_usdt` (Input)
| Campo | Descrição |
|-------|-----------|
| price, amount, side, timestamp, symbol, trade_id |

### `stream:signals:tech` (Output)
| Campo | Descrição |
|-------|-----------|
| timestamp | Timestamp da vela |
| close | Preço de fechamento |
| rsi | RSI(14) |
| macd, macd_signal, macd_hist | MACD(12,26,9) |
| bb_upper, bb_middle, bb_lower | Bollinger(20,2) |
| calc_time_ms | Tempo de cálculo |

## 6. Containers Ativos
| Container | Status | Função |
|-----------|--------|--------|
| vostok_redis | Healthy | Event Bus |
| vostok_timescale | Healthy | Cold Storage |
| vostok_ingestor | Healthy | WebSocket → Redis |
| vostok_quant | Healthy | OHLCV + TA-Lib |

## 7. Diretrizes
- Nunca comitar chaves de API
- Type hints obrigatórios
- Logs JSON estruturados