# 🚀 Operations Quick Start Guide

**Fast-track guide to deploy and operate Noogh Unified System**

---

## ⚡ Quick Deploy Options

### Option 1: Local Development (Fastest)

```bash
# 1. Run all tests
./run.sh test

# 2. Start all services
./run.sh all

# 3. Verify health
curl http://localhost:8000/health

# 4. Access API docs
open http://localhost:8000/docs
```

### Option 2: Docker Compose

```bash
# 1. Build and start
docker-compose up -d

# 2. View logs
docker-compose logs -f api

# 3. Check status
docker-compose ps

# 4. Stop
docker-compose down
```

### Option 3: Kubernetes

```bash
# 1. Deploy everything
kubectl apply -f deploy/kubernetes/

# 2. Check pods
kubectl get pods -n noogh-system -w

# 3. Port-forward
kubectl port-forward svc/noogh-api-internal 8000:8000 -n noogh-system

# 4. Scale
kubectl scale deployment noogh-api --replicas=5 -n noogh-system
```

---

## 📊 Monitoring Setup (5 minutes)

### Step 1: Start Prometheus

```bash
docker run -d \
  -p 9090:9090 \
  -v $(pwd)/deploy/prometheus:/etc/prometheus \
  --name prometheus \
  prom/prometheus
```

### Step 2: Start Grafana

```bash
docker run -d \
  -p 3000:3000 \
  --name grafana \
  grafana/grafana
```

### Step 3: Configure

1. Open Grafana: http://localhost:3000 (admin/admin)
2. Add Prometheus data source: http://prometheus:9090
3. Import dashboard: `deploy/grafana/dashboard.json`

**Done! Full monitoring in 5 minutes.**

---

## 🐕 Enable Self-Healing

### Terminal 1: Start Watchdog

```bash
./scripts/watchdog.sh monitor
```

### Terminal 2: Start Health Monitor

```bash
venv/bin/python scripts/healthcheck.py --monitor
```

**System will auto-restart crashed services!**

---

## 🔔 Alert Testing

```bash
# Trigger high error rate (for testing)
for i in {1..100}; do
  curl http://localhost:8000/nonexistent
done

# Check alerts log
cat logs/alerts.log

# Check Prometheus alerts
open http://localhost:9090/alerts
```

---

## 📈 Key Metrics URLs

- **API Docs:** http://localhost:8000/docs
- **Metrics:** http://localhost:8000/metrics
- **Health:** http://localhost:8000/health
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000

---

## 🔧 Common Commands

```bash
# Check service status
./run.sh status

# View logs
tail -f logs/api.log

# Run tests
./run.sh test

# Stop all
./run.sh stop

# Restart API
./run.sh stop && ./run.sh api
```

---

## 🚨 Emergency Procedures

### Service Won't Start

```bash
# Check ports
lsof -i :8000
lsof -i :8001

# Kill processes
./run.sh stop

# Check logs
tail -100 logs/api.log
tail -100 logs/mcp.log

# Restart
./run.sh all
```

### High Memory Usage

```bash
# Check usage
free -h

# Restart services
./run.sh stop
sleep 5
./run.sh all
```

### Disk Full

```bash
# Clean logs
find logs/ -name "*.log" -mtime +7 -delete

# Clean Docker
docker system prune -a -f

# Clean pip cache
pip cache purge
```

---

## 📞 Support Resources

- **Deployment Report:** [DEVOPS_AUTOMATION_REPORT.md](DEVOPS_AUTOMATION_REPORT.md)
- **Production Guide:** [PRODUCTION_DEPLOYMENT_REPORT.md](PRODUCTION_DEPLOYMENT_REPORT.md)
- **Test Results:** `./run.sh test`
- **Logs:** `logs/` directory

---

**Quick Start Time: <10 minutes**
**Full Stack Time: <30 minutes**

🎉 **Ready to operate!**
