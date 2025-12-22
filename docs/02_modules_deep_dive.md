# 🔬 Modules Deep Dive

> **Technical documentation for each VOSTOK-1 component**  
> **Versão:** 2.0 | **Última Atualização:** 2025-12-22

---

## 📥 INGESTOR

**Location**: `src/ingestor/main.py`  
**Container**: `vostok_ingestor`  
**Purpose**: Real-time market data ingestion from Binance

### Responsibilities
- WebSocket connection management with auto-reconnect
- Trade data parsing (price, quantity, side, timestamp)
- Publishing to Redis Stream `stream:market:btc_usdt`
- Health monitoring and graceful shutdown

### Key Features
```python
# Auto-reconnect with exponential backoff
MAX_RETRIES = 10
RETRY_DELAY = [1, 2, 4, 8, 16, 32, 60, 120, 300, 600]  # seconds

# Trade parsing
{
    "price": "105234.50",
    "quantity": "0.0234",
    "side": "buy",
    "timestamp": 1702831200000,
    "type": "trade"
}
```

### Health Check
- Redis ping every 30s
- WebSocket heartbeat monitoring
- Automatic restart on connection loss

---

## ⚙️ QUANT PROCESSOR

**Location**: `src/quant/main.py`  
**Container**: `vostok_quant`  
**Purpose**: Technical indicator calculation and signal generation

### Indicators Calculated

| Indicator | Window | Description |
|-----------|--------|-------------|
| **RSI** | 14 periods | Relative Strength Index (0-100) |
| **CVD** | Cumulative | Cumulative Volume Delta (buy - sell) |
| **Entropy** | Per candle | Shannon entropy (0-1, market noise) |
| **ATR** | 14 periods | Average True Range (volatility) |
| **MACD** | 12/26/9 | Momentum oscillator |
| **Bollinger** | 20, 2σ | Price bands |

### Dual-Stream Architecture

```
                    ┌────────────────────┐
   Trade Tick ──────│    QUANT           │
                    │    PROCESSOR       │
                    └────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    stream:signals:live           stream:signals:tech
    (Every Tick)                   (Candle Close)
    ──────────────                 ───────────────
    • price                        • OHLCV
    • cvd_current                  • RSI
    • timestamp                    • CVD (final)
    • candle_high/low              • Entropy
                                   • ATR
                                   • MACD
                                   • Funding Rate
```

### Live Pulse vs Closed Candle

| Aspect | Live Pulse | Closed Candle |
|--------|------------|---------------|
| **Frequency** | Every trade (~10-100/sec) | Every minute |
| **Data** | Price, partial CVD | Full indicators |
| **Purpose** | Real-time display | Decision making |
| **Stream** | `stream:signals:live` | `stream:signals:tech` |

---

## 🧠 SENTIMENT (AI Module)

**Location**: `src/sentiment/main.py`  
**Container**: `vostok_sentiment`  
**Purpose**: News sentiment analysis using local LLM

### Architecture

```
RSS Feeds ──► feedparser ──► Headlines ──► Qwen 2.5 ──► Sentiment Score
                                               │
                                               ▼
                                    stream:signals:sentiment
```

### RSS Sources

| Source | URL | Update Frequency |
|--------|-----|------------------|
| CoinDesk | `coindesk.com/arc/outboundfeeds/rss/` | Real-time |
| CoinTelegraph | `cointelegraph.com/rss` | Real-time |
| CryptoPanic | `cryptopanic.com/news/rss/` | Aggregated |

### LLM Configuration

```python
# Ollama API Payload
{
    "model": "qwen2.5:7b-instruct",
    "system": SYSTEM_PROMPT,  # Elite Analyst doctrine
    "options": {
        "temperature": 0.1,   # Maximum precision
        "num_ctx": 4096       # Extended context
    }
}
```

### System Prompt (The Doctrine)

```
ROLE: Elite Crypto Market Analyst (Hedge Fund Tier)

RULES:
1. IGNORE FUD/NOISE → Neutral (0.0)
2. WEIGH REGULATION → SEC/Gov = 2x weight
3. DETECT INSTITUTIONAL FLOW → BlackRock/ETF = High Impact
4. PRIORITIZE FACTS → On-chain > Rumors

SCORING:
• Strong Bullish: +0.8 to +1.0
• Bullish: +0.3 to +0.7
• Neutral: -0.2 to +0.2
• Bearish: -0.7 to -0.3
• Strong Bearish: -1.0 to -0.8
```

### GPU Allocation

- **Device**: NVIDIA RTX 2060 (GPU 0)
- **VRAM**: ~6GB (Qwen 2.5 7B carregado)
- **Keep Alive**: 24h (model stays loaded)

---

## 📊 MONITOR (TUI Dashboard)

**Location**: `src/monitor/main.py`  
**Container**: `vostok_monitor`  
**Purpose**: Real-time terminal dashboard

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│                 VOSTOK-1 SNIPER PROTOCOL  ● ONLINE          │
├────────────────────────────────┬────────────────────────────┤
│      MARKET INTELLIGENCE       │           REGIME           │
│  💰 PRICE    $105,234.50   ▲   │  ENTROPY: 0.4521           │
│  📊 RSI      42.3      NEUTRAL │  ◐ NORMAL                  │
│  📈 CVD      +0.2345   BUYING  │                            │
│  📉 ATR      234.56            │  Signals: 1,234            │
│  📊 MACD     +12.34   BULLISH  │  Pulses: 45,678            │
│  💵 FUNDING  +0.0012%          │  AI: 12                    │
├────────────────────────────────┼────────────────────────────┤
│   🤖 AI SENTIMENT (QWEN 2.5)   │          DATASET           │
│  Score: +0.45 [█████░░░░░]     │  📁 SIZE: 234 lines        │
│  🔥 BULLISH                    │                            │
│  Summary: "Institutional..."   │  RECENT LABELS             │
│  Conf: 85% | Last: 14:30       │  ✓ LONG: +1.24%           │
└────────────────────────────────┴────────────────────────────┘
```

### Multi-Stream Consumption

```python
# 3 concurrent streams
await asyncio.gather(
    self.stream_live_pulse(),    # 50ms block, real-time price
    self.stream_signals(),        # 100ms block, tech indicators
    self.stream_sentiment(),      # 1s block, AI analysis
    self.run_dashboard(),         # 8 FPS refresh
)
```

### State Management

| State Class | Updates From | Persistence |
|-------------|--------------|-------------|
| `MarketState` | Live + Tech streams | Per-tick |
| `SentimentState` | Sentiment stream | 15-min intervals |

---

## 🎯 DECISION ENGINE

**Location**: `src/decision/main.py`  
**Container**: `vostok_decision`  
**Purpose**: Trade signal generation and meta-labeling

### Triple Barrier Method

```
                     TAKE PROFIT (+1.5%)
                          ╱
                         ╱
    ENTRY ──────────────●──────────────────►  TIME LIMIT (60 min)
                         ╲
                          ╲
                     STOP LOSS (-0.75%)
```

### Labeling Logic

| Barrier Hit | Label | Meaning |
|-------------|-------|---------|
| Take Profit | `1` | Successful trade |
| Stop Loss | `0` | Failed trade |
| Time Limit | `0` | No conviction |

---

## 🎓 TRAINER (ML Pipeline)

**Location**: `src/trainer/main.py`  
**Container**: `vostok_trainer`  
**Purpose**: Batch model training

### Pipeline Steps

1. **Ingest**: Load `dataset.jsonl` (min 50 samples)
2. **Prepare**: Extract features (RSI, CVD, Entropy, ATR)
3. **Train**: RandomForest (n=200, depth=10, balanced, no shuffle)
4. **Validate**: Precision, Recall, F1, Confusion Matrix
5. **Export**: Save `sniper_v1.pkl` if precision > 0.35

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,          # Prevent overfitting
    min_samples_leaf=50,   # Require statistical evidence
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,             # Use all cores
)
```

### Current Model Metrics (sniper_v1)

| Metric | Value |
|--------|-------|
| **Precision** | 35.95% |
| **Recall** | 53.77% |
| **F1-Score** | 43.09% |
| **Threshold** | 0.52 |

---

## Next Steps

- [03_strategy_logic.md](./03_strategy_logic.md) - Trading Strategy
- [04_operations_manual.md](./04_operations_manual.md) - Operations Guide
