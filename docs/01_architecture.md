# 🏗️ VOSTOK-1 Architecture Overview

> **Hybrid Trading System: Algorithmic + Generative AI**

## System Philosophy

VOSTOK-1 is a **Hybrid High-Frequency Trading (HFT) System** that combines:
- 🤖 **Algorithmic Analysis**: Real-time technical indicators (RSI, CVD, ATR, Entropy)
- 🧠 **Generative AI**: Sentiment analysis via local LLM (Qwen 2.5)
- 📊 **Machine Learning**: Meta-labeling with Random Forest (training pipeline)

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph External["🌐 External Data Sources"]
        BINANCE["Binance WebSocket<br/>BTC/USDT Trades"]
        RSS["RSS Feeds<br/>CoinDesk/CoinTelegraph"]
    end

    subgraph Ingestion["📥 Data Ingestion Layer"]
        INGESTOR["INGESTOR<br/>WebSocket Manager<br/>Auto-Reconnect"]
    end

    subgraph Processing["⚙️ Processing Layer"]
        QUANT["QUANT PROCESSOR<br/>RSI, CVD, ATR, Entropy<br/>MACD, Bollinger"]
        SENTIMENT["SENTIMENT<br/>RSS Parser + Qwen LLM<br/>GPU Accelerated"]
        OLLAMA["OLLAMA<br/>Qwen 2.5 (7B)<br/>GPU 0 Dedicated"]
    end

    subgraph Storage["💾 Storage Layer"]
        REDIS["REDIS<br/>Streams + Pub/Sub"]
        TIMESCALE["TimescaleDB<br/>Historical Data"]
        DATASET["dataset.jsonl<br/>Training Data"]
        MODEL["sniper_v1.pkl<br/>ML Model"]
    end

    subgraph Consumption["📊 Consumption Layer"]
        MONITOR["MONITOR TUI<br/>Rich Dashboard<br/>8 FPS Refresh"]
        DECISION["DECISION ENGINE<br/>Triple Barrier<br/>Meta-Labeling"]
        TRAINER["TRAINER<br/>Random Forest<br/>Batch Processing"]
    end

    %% Data Flow
    BINANCE -->|trades| INGESTOR
    INGESTOR -->|stream:market:btc_usdt| REDIS
    
    REDIS -->|market data| QUANT
    QUANT -->|stream:signals:tech| REDIS
    QUANT -->|stream:signals:live| REDIS
    
    RSS -->|headlines| SENTIMENT
    SENTIMENT -->|prompt| OLLAMA
    OLLAMA -->|JSON response| SENTIMENT
    SENTIMENT -->|stream:signals:sentiment| REDIS
    
    REDIS -->|tech + live + sentiment| MONITOR
    REDIS -->|signals| DECISION
    DECISION -->|labeled trades| DATASET
    
    DATASET -->|training data| TRAINER
    TRAINER -->|trained model| MODEL

    %% Styling
    classDef external fill:#1a1a2e,stroke:#16213e,color:#fff
    classDef ingestion fill:#0f3460,stroke:#16213e,color:#fff
    classDef processing fill:#533483,stroke:#16213e,color:#fff
    classDef storage fill:#e94560,stroke:#16213e,color:#fff
    classDef consumption fill:#16a085,stroke:#16213e,color:#fff

    class BINANCE,RSS external
    class INGESTOR ingestion
    class QUANT,SENTIMENT,OLLAMA processing
    class REDIS,TIMESCALE,DATASET,MODEL storage
    class MONITOR,DECISION,TRAINER consumption
```

---

## Data Streams (Redis)

| Stream | Producer | Consumer(s) | Content |
|--------|----------|-------------|---------|
| `stream:market:btc_usdt` | Ingestor | Quant | Raw trades (price, volume, side) |
| `stream:signals:tech` | Quant | Monitor, Decision | Full candle + indicators |
| `stream:signals:live` | Quant | Monitor | Real-time price/CVD pulse |
| `stream:signals:sentiment` | Sentiment | Monitor, Decision | AI sentiment score |

---

## Container Topology

```
┌─────────────────────────────────────────────────────────────┐
│                     VOSTOK-1 CLUSTER                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  REDIS  │  │TIMESCALE│  │ OLLAMA  │  │INGESTOR │        │
│  │  :6379  │  │  :5432  │  │ :11434  │  │ WS/REST │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│       └────────────┼────────────┼────────────┘              │
│                    │            │                           │
│  ┌─────────┐  ┌────┴────┐  ┌───┴─────┐  ┌─────────┐        │
│  │  QUANT  │  │DECISION │  │SENTIMENT│  │ MONITOR │        │
│  │Processor│  │ Engine  │  │  (LLM)  │  │  (TUI)  │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│                                                             │
│  ┌─────────┐                                               │
│  │ TRAINER │  (Batch Job - On Demand)                      │
│  │  (ML)   │                                               │
│  └─────────┘                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.11 | Core services |
| **Message Bus** | Redis Streams | Real-time event streaming |
| **Database** | TimescaleDB | Time-series historical data |
| **LLM** | Ollama + Qwen 2.5 (7B) | Local sentiment analysis |
| **ML** | scikit-learn (Random Forest) | Meta-labeling |
| **TUI** | Rich | Terminal dashboard |
| **Container** | Docker + Compose | Orchestration |

---

## Network Configuration

- **Network**: `vostok_net` (bridge mode)
- **Internal DNS**: Containers communicate via service names
- **Exposed Ports**:
  - Redis: 6379
  - TimescaleDB: 5432
  - Ollama API: 11434

---

## GPU Allocation

| GPU | Device ID | Allocation |
|-----|-----------|------------|
| Quadro P2000 | `0` | Ollama/Qwen (Dedicated) |
| Quadro P2000 | `1` | Desktop/Windows |

---

## Next Steps

- [02_modules_deep_dive.md](./02_modules_deep_dive.md) - Component Details
- [03_strategy_logic.md](./03_strategy_logic.md) - Trading Strategy
- [04_operations_manual.md](./04_operations_manual.md) - Operations Guide
