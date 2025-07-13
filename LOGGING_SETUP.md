# MLflow Logging with ELK Stack Setup

This guide explains how to set up comprehensive logging for your MLflow pipeline using the ELK (Elasticsearch, Logstash, Kibana) stack.

## Overview

The logging system provides:
- **Structured logging** for MLflow pipeline events
- **Real-time monitoring** through Kibana dashboards
- **Error tracking** and analysis
- **Performance metrics** visualization
- **Integration** with existing ELK stack

## Fixed Issues

### 1. Permission Error Resolution
- **Problem**: `PermissionError: [Errno 13] Permission denied: '/home/nourbayoudh/MLOps/mlflow.log'`
- **Solution**: Updated `logging_config.py` to:
  - Create logs in a local `logs/` directory
  - Handle permission errors gracefully
  - Fallback to console logging if file creation fails

### 2. Enhanced Logging Structure
- Added structured JSON logging for MLflow events
- Implemented comprehensive error handling
- Added detailed event tracking throughout the pipeline

## File Structure

```
MLOps/
├── logs/                    # Log files directory
│   └── mlflow.log         # MLflow pipeline logs
├── main.py                 # Updated main script with logging
├── logging_config.py       # Enhanced logging configuration
├── logstash/
│   └── config/
│       └── pipeline/
│           └── logstash.conf  # Logstash pipeline configuration
├── kibana/
│   └── config/
│       └── dashboard.json     # Kibana dashboard configuration
└── setup/
    └── import_kibana_dashboard.sh  # Dashboard import script
```

## Key Features

### 1. Structured Event Logging
The system logs structured events in JSON format:
```json
{
  "timestamp": "2023-07-13T23:55:15.123456",
  "event_type": "model_training_start",
  "details": {
    "model_type": "GradientBoostingRegressor",
    "hyperparams": {...},
    "train_samples": 1000
  },
  "source": "mlflow_pipeline"
}
```

### 2. Event Types Tracked
- `data_loading_start/success/error`
- `data_cleaning_start/success/error`
- `data_preparation_start/success/error`
- `scaling_start/success/error`
- `model_training_start/success/error`
- `model_evaluation_start/success/error`
- `model_saving_start/success/error`
- `mlflow_run_start/success/error/end`

### 3. Logstash Pipeline
The Logstash configuration:
- Parses MLflow event logs
- Extracts structured data
- Sends to Elasticsearch with proper indexing
- Handles both structured and general logs

### 4. Kibana Dashboard
The dashboard includes:
- **Events Overview**: Timeline of all MLflow events
- **Metrics Timeline**: RMSE and R² metrics over time
- **Event Types Distribution**: Pie chart of event categories
- **Error Analysis**: Table of errors and their frequency
- **Pipeline Steps Performance**: Bar chart of step performance

## Usage

### 1. Run the MLflow Pipeline
```bash
python3 main.py
```

This will:
- Create logs in `logs/mlflow.log`
- Generate structured events for each pipeline step
- Handle errors gracefully with detailed logging

### 2. Start the ELK Stack
```bash
# Start Elasticsearch, Logstash, and Kibana
docker-compose up -d
```

### 3. Import Kibana Dashboard
```bash
# Make the script executable
chmod +x setup/import_kibana_dashboard.sh

# Run the import script
./setup/import_kibana_dashboard.sh
```

### 4. Access the Dashboard
Open your browser and go to:
```
http://localhost:5601/app/kibana#/dashboard/mlflow-dashboard
```

## Log Analysis

### 1. Real-time Monitoring
- Monitor pipeline execution in real-time
- Track performance metrics (RMSE, R²)
- Identify bottlenecks and errors

### 2. Error Analysis
- View error frequency and types
- Analyze error patterns
- Debug pipeline issues

### 3. Performance Insights
- Track model training time
- Monitor data processing steps
- Analyze resource usage

## Configuration

### Logstash Configuration
The Logstash pipeline is configured in `logstash/config/pipeline/logstash.conf`:
- Inputs: Filebeat, direct file input, heartbeat
- Filters: Parse MLflow events, extract metadata
- Outputs: Elasticsearch with proper indexing

### Kibana Dashboard
The dashboard configuration is in `kibana/config/dashboard.json`:
- 5 different visualizations
- Real-time data updates
- Interactive filtering and drill-down

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - The system now handles permission errors gracefully
   - Falls back to console logging if needed

2. **Logstash Connection Issues**
   - Ensure Elasticsearch is running
   - Check network connectivity between containers

3. **Dashboard Not Loading**
   - Wait for Kibana to fully start
   - Check if the dashboard import was successful

### Debug Commands

```bash
# Check if logs are being generated
tail -f logs/mlflow.log

# Check Logstash logs
docker logs logstash

# Check Elasticsearch status
curl http://localhost:9200/_cluster/health

# Check Kibana status
curl http://localhost:5601/api/status
```

## Advanced Features

### 1. Custom Event Logging
Add custom events in your code:
```python
log_mlflow_event("custom_event", {
    "custom_field": "value",
    "metric": 42.0
})
```

### 2. Filtering in Kibana
Use Kibana's query language to filter logs:
- `pipeline:mlflow` - Show only MLflow logs
- `event_category:model_training_error` - Show training errors
- `mlflow_event.details.rmse > 0.5` - Show high RMSE runs

### 3. Alerts
Set up Kibana alerts for:
- High error rates
- Poor model performance
- Pipeline failures

## Performance Considerations

- Log files are rotated automatically
- Elasticsearch indexing is optimized for time-series data
- Dashboard queries are cached for better performance

## Security

- Logs contain sensitive model information
- Ensure proper access controls
- Consider encryption for production environments

## Next Steps

1. **Customize Dashboard**: Modify the dashboard for your specific needs
2. **Add Alerts**: Set up Kibana alerts for critical events
3. **Scale**: Consider using Elasticsearch clusters for production
4. **Integrate**: Connect with other monitoring tools (Prometheus, Grafana) 