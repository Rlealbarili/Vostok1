# AGENTS.MD - Vostok-1 Knowledge Base

> **Status do Sistema:** 🟢 DECISION ENGINE OPERACIONAL
> **Última Atualização:** 2024-12-17T09:45:00-03:00
> **Engenheiro Chefe:** Petrovich
> **Operador:** Vostok

## 1. Missão
Sistema de trading autônomo com Data Labeling para treino de Meta-Labeling.

## 2. Arquitetura
```
Binance WS → Ingestor → stream:market → Quant → stream:signals:tech → Decision
                                                                        ↓
                                                          data/training_dataset.jsonl
```

## 3. Estado Atual
- [x] Infraestrutura Docker (Redis 7 + TimescaleDB PG16)
- [x] **Ingestor Sniper**: watch_trades + watch_funding_rate ✅
- [x] **Quant Sniper**: CVD + Entropia + ATR + Parkinson ✅
- [x] **Decision Engine**: StrategyEngine + TripleBarrierLabeler ✅
- [ ] Módulo Sentiment (Qwen)
- [ ] Módulo Executor

## 4. Decision Engine (2024-12-17) ✅

### StrategyEngine (Sinais Base)
- BUY: `RSI < 35` E `CVD > 0` (divergência bullish)
- SELL: `RSI > 65` E `CVD < 0` (divergência bearish)
- Cooldown: 5 velas entre sinais

### TripleBarrierLabeler
- Take Profit: `ATR * 2.0`
- Stop Loss: `ATR * 1.0`
- Tempo máximo: 120 velas (2h)
- Labels: `1` (WIN) ou `0` (LOSS)

### Dataset Output
```
data/training_dataset.jsonl
```

## 5. Containers
| Container | Status | Função |
|-----------|--------|--------|
| vostok_redis | Healthy | Event Bus |
| vostok_timescale | Healthy | Cold Storage |
| vostok_ingestor | Healthy | Trades + Funding |
| vostok_quant | Healthy | OHLCV + Indicators |
| vostok_decision | Healthy | Data Labeling |

## 6. Streams Redis
| Stream | Conteúdo |
|--------|----------|
| stream:market:btc_usdt | Trades + Funding rates |
| stream:signals:tech | OHLCV + CVD + ATR + RSI/MACD |

## 7. Diretrizes
- Nunca comitar chaves de API
- Type hints obrigatórios
- Logs JSON estruturados