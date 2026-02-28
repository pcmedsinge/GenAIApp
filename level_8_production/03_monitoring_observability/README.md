# Project 3: Monitoring & Observability

## What You'll Learn
- Build LLM observability from scratch (no vendor lock-in)
- Track every request: tokens, cost, latency, errors
- Build cost dashboards and budget alerts
- Trace agent execution step-by-step

## Why Monitoring?
```
Without monitoring:  "Is it working?" → "I think so?" → 💸💸💸
With monitoring:     Request logs, cost tracking, latency P95, error rates, alerts
```

In production, you MUST know: how much you're spending, how fast responses are,
and when things break.

## Running the Code
```bash
cd level_8_production/03_monitoring_observability
python main.py
```

## Demos
1. **Request Logger** — Log every LLM call with timestamp, tokens, cost, latency
2. **Cost Tracker** — Track spend per model, per endpoint, per day
3. **Latency Monitor** — P50/P95/P99 latency, anomaly detection, alerts
4. **Agent Tracer** — Step-by-step trace of agent interactions with timing

## Exercises
1. Build comprehensive request logger (JSONL storage, queryable)
2. Build text-based cost dashboard with budget alerts
3. Build agent execution tracer with tree visualization
4. Build alerting system for latency spikes and cost anomalies

## Key Concepts
- **Structured logging**: JSON logs with consistent fields
- **Percentile latency**: P50 (median), P95, P99 for SLA tracking
- **Token economics**: Input vs output tokens, per-model pricing
- **Distributed tracing**: Following a request through agent → tool → LLM calls
