# Cloud evaluation worksheet

Fill every cell from your own run on the deployed service. Leave nothing estimated: the course requires
measured values. Record the date, region and configuration used.

**Configuration under test:** region ____ · CPU/memory ____ · concurrency per instance ____ · min/max instances ____

## Experiment 1 - Response time at increasing concurrency

Command: `python loadtest/measure.py <url> --requests 200 --concurrency <C>` (run each twice; use the second run to
exclude the cold start, and report the cold start separately).

| Concurrency | Median (ms) | p95 (ms) | Max (ms) | Throughput (req/s) | Errors | Instances observed |
|---|---|---|---|---|---|---|
| 1  |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 20 |  |  |  |  |  |  |
| 50 |  |  |  |  |  |  |

Cold-start time (first request after the service was idle): ____ ms

## Experiment 2 - Autoscaling

Locust: ramp to ____ users over ____ minutes. Read the instance count from Cloud Run → Metrics.

| Time into test | Active users | Container instances | p95 latency (ms) |
|---|---|---|---|
|  |  |  |  |

## Experiment 3 - Durability of the history

1. Save 3 predictions. 2. Deploy a new revision (or wait until the instance restarts). 3. Open History.
Result with `HISTORY_BACKEND=csv`: ____ · Result with `HISTORY_BACKEND=firestore`: ____

## Experiment 4 - Cost

Cost shown by the billing report for the test period: ____ (state the free-tier allowance used).

## Cloud concepts to link in the report (choose 4 to 6 that match your lectures)

Elasticity and auto-scaling · Pay-per-use pricing · Containerisation · Managed services (PaaS) ·
Statelessness and durable storage · High availability across instances · Shared-responsibility security model.
