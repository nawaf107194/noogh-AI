# 📊 Noogh Unified System - Grafana Monitoring Dashboard

## ✅ Complete Setup Guide

This guide will help you import and configure the complete Grafana monitoring dashboard for the Noogh Unified System.

---

## 🎯 What You'll Get

A comprehensive real-time monitoring dashboard with:

### System Overview Section
- **System Health Gauge** (0-100%) with color thresholds
- **Active Components** stat
- **Total Components** stat
- **Components Breakdown** pie chart
- **Component Status** multi-stat panel

### Ministers Section (الوزراء)
- **All Ministers Table** - Full list of 14 ministers with status
- **Active Ministers Gauge** - Visual representation of active count
- **Total Ministers** stat
- **Ministers by Domain** bar chart

### Brain v4.0 Intelligence Section
- **Brain Status** - Active/Inactive indicator
- **Brain Memory Usage** gauge (0-100%)
- **Session Memories** count with trend
- **Brain Capacity** gauge

### Knowledge Index & MCP Section
- **Knowledge Progress** gauge (target: 100 chunks)
- **MCP Server Status** - Active/Inactive indicator
- **MCP Resources** - Tools and Resources count
- **MCP Statistics** - Uptime and Total Requests

### Training & Automation Section
- **Daily Training Status** - Success/Failed indicator
- **Training Tasks** - Completed vs Total
- **Training History** - Total runs and successful runs
- **Cron Automation** - Active status with schedule

### Recent Logs Section
- **System Logs Table** - Last 100 log entries with:
  - Timestamp
  - Level (color-coded: INFO, WARNING, ERROR, SUCCESS)
  - Source component
  - Message

---

## 📋 Prerequisites

### 1. Grafana with Infinity Plugin
```bash
# Already installed from previous steps!
sudo grafana-cli plugins ls | grep infinity
```

You should see:
```
yesoreyeram-infinity-datasource @ 3.6.0
```

### 2. API Server Running
```bash
# Check if API server is running on port 8000
curl -s http://localhost:8000/api/system/health | jq '.health_percent'
```

Should return: `80.0` (or current health percentage)

### 3. Data Source Configured
The dashboard expects a data source named **"Noogh System API"** with UID: `noogh_api`

---

## 🚀 Step-by-Step Import Instructions

### Step 1: Access Grafana

1. Open browser: `http://localhost:3000`
2. Login with your Grafana credentials

### Step 2: Configure Data Source (If Not Done)

1. Go to **⚙️ Administration** → **Data sources**
2. Click **Add data source**
3. Search for and select **"Infinity"**
4. Configure:
   - **Name**: `Noogh System API`
   - **URL**: `http://localhost:8000`
5. Click **Save & test**
6. Verify you see: ✅ **Health check successful**

**Important**: Edit the data source and check its UID:
- Click on the data source name
- Look at the URL in your browser
- Note the UID (e.g., `/datasources/edit/df4ga7s13zfgge`)
- If it's NOT `noogh_api`, you'll need to update the dashboard JSON

### Step 3: Import the Dashboard

1. In Grafana, click **+** (plus icon) in the left sidebar
2. Select **Import dashboard**
3. You have two options:

**Option A: Upload JSON file**
```bash
# The dashboard file is located at:
/home/noogh/projects/noogh_unified_system/grafana/noogh_unified_system_dashboard.json
```
- Click **Upload JSON file**
- Navigate to the file above and select it

**Option B: Paste JSON**
- Copy the contents of the JSON file
- Paste into the **Import via panel json** text box

4. Click **Load**

5. On the import screen:
   - **Name**: Keep as "Noogh Unified System - Complete Monitoring Dashboard" or customize
   - **Folder**: Select a folder or leave as "General"
   - **UID**: Keep as `noogh_unified_dashboard` or customize
   - **Data source**: If prompted, select "Noogh System API"

6. Click **Import**

### Step 4: Verify Dashboard

After import, you should see:
- ✅ System Health gauge showing **80%**
- ✅ Active Components showing **4**
- ✅ Ministers table with **14 ministers** all active
- ✅ Knowledge progress showing **89%** (89/100 chunks)
- ✅ Brain status showing **Active v4.0**
- ✅ System logs table populated with recent entries

---

## 🔄 Dashboard Auto-Refresh

The dashboard is configured to **auto-refresh every 5 seconds**.

You can change this:
1. Click the **🕐 refresh icon** in the top-right
2. Select your preferred interval: 5s, 10s, 30s, 1m, 5m, etc.

---

## 🎨 Customization

### Change Data Source UID (if needed)

If your data source has a different UID than `noogh_api`:

1. Open the dashboard JSON file in an editor
2. Find and replace all instances of:
   ```json
   "uid": "noogh_api"
   ```
   With your actual UID:
   ```json
   "uid": "your_actual_uid"
   ```
3. Re-import the dashboard

### Modify Panel Queries

Each panel queries specific endpoints:
- `/api/system/metrics` - Main metrics endpoint (23 metrics)
- `/api/system/ministers/metrics` - Ministers data (14 ministers)
- `/api/system/logs/recent` - Recent logs (last 100 entries)

To modify a panel:
1. Click the panel title
2. Select **Edit**
3. Modify the **URL** field under the query
4. Click **Run Query** to test
5. Click **Apply** to save

### Add New Panels

To add custom panels:
1. Click **Add** → **Visualization**
2. Select **Noogh System API** as data source
3. Configure query:
   - **Type**: JSON
   - **Parser**: Backend
   - **Source**: URL
   - **URL**: Choose an endpoint (e.g., `/api/system/metrics`)
4. Add **Transformations** to filter/format data:
   - **Filter by value** - Filter specific metrics
   - **Organize** - Rename/exclude fields
   - **Convert field type** - Convert timestamps
5. Choose visualization type: Stat, Gauge, Table, Time series, etc.
6. Configure display options
7. Click **Apply**

---

## 📊 Available API Endpoints

### Metrics Endpoints

1. **`/api/system/metrics`** - Prometheus-style metrics
   ```bash
   curl http://localhost:8000/api/system/metrics | jq '.[0:3]'
   ```
   Returns 23 metrics including:
   - `system_health_percent`
   - `active_components`
   - `brain_memories_count`
   - `knowledge_chunks_total`
   - `ministers_active`
   - And more...

2. **`/api/system/ministers/metrics`** - Individual minister data
   ```bash
   curl http://localhost:8000/api/system/ministers/metrics | jq '.[0:2]'
   ```
   Returns 14 ministers with:
   - `minister_id`
   - `minister_name`
   - `domain`
   - `status`
   - `status_numeric` (1 = active, 0 = inactive)

3. **`/api/system/logs/recent`** - Recent system logs
   ```bash
   curl http://localhost:8000/api/system/logs/recent | jq '. | length'
   ```
   Returns last 100 log entries with:
   - `timestamp`
   - `level` (INFO, WARNING, ERROR, SUCCESS)
   - `source` (component name)
   - `message`

4. **`/api/system/metrics/timeseries`** - Time-series data
   ```bash
   curl http://localhost:8000/api/system/metrics/timeseries | jq '.'
   ```
   Returns datapoints for graphing over time

### Legacy Endpoints (Also Available)

- `/api/system/overview` - Complete system state
- `/api/system/health` - Quick health check
- `/api/system/brain` - Brain v4.0 details
- `/api/system/knowledge` - Knowledge index stats
- `/api/system/ministers` - Ministers with summary
- `/api/system/mcp` - MCP server details
- `/api/system/training/history` - Training reports

---

## 🔍 Troubleshooting

### Issue: Panels show "No data"

**Solution 1**: Check API server
```bash
curl http://localhost:8000/api/system/health
```
If error, restart API:
```bash
cd /home/noogh/projects/noogh_unified_system
export PYTHONPATH=/home/noogh/projects/noogh_unified_system/src
./venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Solution 2**: Check data source connection
- Go to **Data sources** → **Noogh System API**
- Click **Save & test**
- Should show: ✅ Health check successful

**Solution 3**: Check panel query
- Edit the panel
- Click **Run Query**
- Check for errors in query inspector

### Issue: "Data source not found"

**Solution**: Update dashboard data source UID
1. Get your data source UID from the URL when editing it
2. Update the dashboard JSON file
3. Re-import the dashboard

### Issue: Logs panel empty

**Solution**: Check logs endpoint
```bash
curl http://localhost:8000/api/system/logs/recent | jq '. | length'
```

If returns 0 or error:
- Verify log files exist in project
- Check system_status.py `get_logs_summary()` function
- Ensure API server has read access to log files

### Issue: Ministers table shows wrong count

**Solution**: Verify ministers data
```bash
curl http://localhost:8000/api/system/ministers/metrics | jq '. | length'
```
Should return: `14`

---

## 📸 Expected Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                        System Overview                           │
├──────────────┬──────────┬──────────┬────────────┬───────────────┤
│ System       │ Active:4 │ Total:5  │ Components │  Component    │
│ Health: 80%  │          │          │ Breakdown  │   Status      │
│   [GAUGE]    │  [STAT]  │  [STAT]  │    [PIE]   │   [STATS]     │
├──────────────┴──────────┴──────────┴────────────┴───────────────┤
│                    Ministers (الوزراء)                           │
├─────────────────────────────────────┬──────────┬────────────────┤
│                                     │  Active  │     Total      │
│   All Ministers Status              │  14/14   │      14        │
│          [TABLE]                    │ [GAUGE]  │    [STAT]      │
│                                     ├──────────┴────────────────┤
│                                     │  Ministers by Domain       │
│                                     │      [BAR CHART]           │
├─────────────────────────────────────┴────────────────────────────┤
│                     Brain v4.0 Intelligence                      │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ Brain Status │ Memory Usage │   Session    │   Brain Capacity   │
│ Active v4.0  │     1%       │  Memories:1  │       100          │
│   [STAT]     │   [GAUGE]    │   [STAT]     │     [GAUGE]        │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                   Knowledge Index & MCP                          │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Knowledge   │ MCP Server   │ MCP Resources│  MCP Statistics    │
│  Progress    │ Active v2.0  │ Tools: 8     │  Uptime: 0s        │
│   89/100     │   [STAT]     │ Resources: 4 │  Requests: 0       │
│   [GAUGE]    │              │   [STATS]    │     [STATS]        │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                    Training & Automation                         │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│   Training   │   Training   │   Training   │      Cron          │
│    Status    │     Tasks    │   History    │   Automation       │
│   Success    │     4/4      │  Runs: 2/0   │ Active (2AM Daily) │
│   [STAT]     │   [STATS]    │   [STATS]    │      [STAT]        │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                         Recent Logs                              │
├──────────────────────────────────────────────────────────────────┤
│ Time         Level    Source     Message                         │
│ 18:50:00    INFO     api        Server started...                │
│ 18:49:45    SUCCESS  training   Daily training completed         │
│ 18:49:30    WARNING  mcp        MCP server inactive              │
│                      [TABLE - Last 100 entries]                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🎉 Success Criteria

After successful import, verify:

- ✅ **24 panels** displaying data
- ✅ **System Health** showing 80%
- ✅ **14 Ministers** in table, all active
- ✅ **Brain v4.0** showing active status
- ✅ **Knowledge** showing 89/100 chunks (89%)
- ✅ **Logs** table populated with entries
- ✅ **Auto-refresh** working every 5 seconds
- ✅ **No "No data" or error messages**

---

## 📚 Next Steps

### 1. Set Up Alerting (Optional)

Configure alerts for critical metrics:
- System health drops below 60%
- Brain becomes inactive
- Training fails
- MCP server goes down

### 2. Create Additional Dashboards

- **Training Dashboard**: Detailed training metrics and history
- **Ministers Dashboard**: Individual minister performance
- **Logs Dashboard**: Advanced log filtering and analysis
- **Performance Dashboard**: API response times, resource usage

### 3. Export & Share

Export the dashboard:
1. Click **Share** icon (top-right)
2. Select **Export** tab
3. Click **Save to file**
4. Share JSON with team members

### 4. Schedule Reports (Grafana Pro)

If using Grafana Pro/Enterprise:
- Set up daily/weekly email reports
- Create PDF snapshots
- Share with stakeholders

---

## 🆘 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Verify all prerequisites are met
3. Test API endpoints directly with `curl`
4. Check Grafana and API server logs

---

## 📝 Dashboard Specifications

- **Grafana Version**: 12.2.1+
- **Plugin Required**: Infinity v3.6.0+
- **Panels**: 24 visualizations
- **Rows**: 6 sections
- **Refresh Rate**: 5 seconds (configurable)
- **Time Range**: Last 6 hours (configurable)
- **Tags**: noogh, unified-system, ai, monitoring

---

**Dashboard UID**: `noogh_unified_dashboard`
**Dashboard Version**: 1.0
**Created**: 2025-11-18
**Author**: Noogh Unified System - Autonomous AI

**File Location**: `/home/noogh/projects/noogh_unified_system/grafana/noogh_unified_system_dashboard.json`

---

🎯 **Ready to import!** Follow the steps above and you'll have a fully functional monitoring dashboard in minutes.
