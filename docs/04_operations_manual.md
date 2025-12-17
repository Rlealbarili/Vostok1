# 📖 Operations Manual

> **Day-to-day operation guide for VOSTOK-1**

---

## Quick Reference

### Essential Commands

```bash
# Start all services
docker compose up -d

# View live dashboard
docker compose run --rm monitor

# Stop all services
docker compose down

# Full restart with rebuild
docker compose down --remove-orphans
docker compose up -d --build
```

---

## Service Management

### Starting Services

```bash
# Start core infrastructure
docker compose up -d redis timescaledb

# Start data pipeline
docker compose up -d ingestor quant decision

# Start AI services (requires GPU)
docker compose up -d llm_engine sentiment

# Start monitoring
docker compose up -d monitor
```

### Checking Status

```bash
# View all containers
docker ps --format "table {{.Names}}\t{{.Status}}"

# Expected output:
# NAMES                STATUS
# vostok_llm           Up X min (healthy)
# vostok_sentiment     Up X min
# vostok_monitor       Up X min
# vostok_decision      Up X min (healthy)
# vostok_quant         Up X min (healthy)
# vostok_ingestor      Up X min (healthy)
# vostok_redis         Up X hours (healthy)
# vostok_timescale     Up X hours (healthy)
```

### Viewing Logs

```bash
# Real-time logs for specific service
docker logs -f vostok_ingestor
docker logs -f vostok_quant
docker logs -f vostok_sentiment
docker logs -f vostok_decision

# Last N lines
docker logs --tail 50 vostok_sentiment

# All services
docker compose logs -f
```

---

## Troubleshooting

### Common Issues

#### 1. Ingestor Not Connecting

```bash
# Check logs
docker logs vostok_ingestor --tail 20

# Verify API keys in .env
cat .env | grep BINANCE

# Restart
docker compose restart ingestor
```

#### 2. Sentiment Not Analyzing

```bash
# Check if LLM is healthy
docker ps | grep llm

# Check sentiment logs
docker logs vostok_sentiment --tail 30

# Restart LLM + Sentiment
docker compose restart llm_engine sentiment
```

#### 3. Monitor Shows OFFLINE

```bash
# Check Redis connection
docker exec vostok_redis redis-cli ping
# Should return: PONG

# Check if streams have data
docker exec vostok_redis redis-cli XLEN stream:signals:tech
```

#### 4. Docker Build Fails (500 Error)

Docker Hub occasionally has outages. Wait 10-30 minutes and retry:

```bash
docker compose build --no-cache <service>
```

---

## Hardware Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| **CPU** | 4 cores |
| **RAM** | 8 GB |
| **GPU** | NVIDIA (4GB VRAM) |
| **Storage** | 20 GB SSD |

### Recommended

| Component | Requirement |
|-----------|-------------|
| **CPU** | 8+ cores |
| **RAM** | 16 GB |
| **GPU** | NVIDIA (8GB VRAM) |
| **Storage** | 50 GB NVMe |

### GPU Configuration

The LLM runs on **GPU 0** (dedicated). If you have only one GPU:

```yaml
# In docker-compose.yml, comment out GPU reservation:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           device_ids: ['0']
#           capabilities: [gpu]
```

---

## Data Management

### File Locations

| File/Directory | Purpose | Backup Priority |
|----------------|---------|-----------------|
| `data/training/dataset.jsonl` | Training data | ⭐⭐⭐ HIGH |
| `models/sniper_v1.pkl` | Trained model | ⭐⭐ MEDIUM |
| `.env` | Configuration | ⭐⭐⭐ HIGH |
| `docker-compose.yml` | Orchestration | ⭐ LOW (in Git) |

### Backup Commands

```bash
# Backup training data
cp data/training/dataset.jsonl data/training/dataset.jsonl.bak

# Backup models
cp -r models/ models.bak/

# Full backup
tar -czvf vostok_backup_$(date +%Y%m%d).tar.gz data/ models/ .env
```

### Dataset Inspection

```bash
# Count training samples
wc -l data/training/dataset.jsonl

# View last 5 entries
tail -5 data/training/dataset.jsonl | jq .

# Check for valid JSON
cat data/training/dataset.jsonl | jq . > /dev/null && echo "Valid JSON"
```

---

## Training the ML Model

```bash
# Ensure you have enough data (min 50 samples)
wc -l data/training/dataset.jsonl

# Run trainer
docker compose run --rm trainer

# Check output
ls -la models/
```

### Expected Trainer Output

```
╔══════════════════════════════════════════════════════════════╗
║   VOSTOK-1 :: MODEL TRAINER PIPELINE                        ║
╚══════════════════════════════════════════════════════════════╝

STEP 1: Loading dataset... ✅ 234 samples
STEP 2: Preparing features... ✅ 5 features
STEP 3: Training RandomForest... ✅
STEP 4: Validating model...
  Precision: 0.67
  Recall: 0.58
  F1-Score: 0.62
STEP 5: Exporting model... ✅ sniper_v1.pkl saved
```

---

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `BINANCE_API_KEY` | Binance API key | `abc123...` |
| `BINANCE_API_SECRET` | Binance API secret | `xyz789...` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `SYMBOL` | `BTC/USDT` | Trading pair |
| `ANALYSIS_INTERVAL` | `900` | Sentiment interval (seconds) |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | Ollama model |

---

## Monitoring Dashboard

### Launching

```bash
# Interactive mode (recommended)
docker compose run --rm monitor

# Background mode
docker compose up -d monitor
docker logs -f vostok_monitor
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Exit dashboard |
| `q` | Quit (if supported) |

### Panel Descriptions

| Panel | Content |
|-------|---------|
| **MARKET INTELLIGENCE** | Price, RSI, CVD, ATR, MACD, Funding |
| **AI SENTIMENT** | Score (-1 to +1), Summary, Confidence |
| **REGIME** | Entropy, Signal/Pulse counters |
| **DATASET** | Training data size, recent labels |

---

## Maintenance Schedule

### Daily

- [ ] Check container health: `docker ps`
- [ ] Review sentiment logs for API errors
- [ ] Verify Redis stream lengths

### Weekly

- [ ] Backup `dataset.jsonl`
- [ ] Review training data quality
- [ ] Check disk space: `df -h`

### Monthly

- [ ] Retrain ML model if data > 500 samples
- [ ] Update Docker images: `docker compose pull`
- [ ] Review and optimize strategy parameters

---

## Emergency Procedures

### Full System Restart

```bash
# Stop everything
docker compose down --remove-orphans

# Clear Redis (if needed)
docker volume rm vostok_redis_data

# Rebuild and start
docker compose up -d --build
```

### LLM Not Responding

```bash
# Restart Ollama
docker compose restart llm_engine

# Check GPU usage
nvidia-smi

# Reload model
docker exec vostok_llm ollama run qwen2.5:7b-instruct "test"
```

### Data Recovery

```bash
# Restore backup
cp data/training/dataset.jsonl.bak data/training/dataset.jsonl

# Or from tar backup
tar -xzvf vostok_backup_20231217.tar.gz
```

---

## Support

- **GitHub**: [Vostok1 Repository](https://github.com/Rlealbarili/Vostok1)
- **Logs**: Always include output of `docker logs <container>`
- **Configuration**: Never share `.env` publicly
