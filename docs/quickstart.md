# Quick Start Guide

Get up and running with the Balancer platform in minutes! This guide shows you the fastest way to start optimizing your supply chain operations.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/balancer.git
cd balancer

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python -m pytest tests/ -v
```

## 5-Minute Examples

### 1. Demand Forecasting (2 minutes)

```python
from demand.service import DemandService
import pandas as pd

# Sample demand data (last 30 days)
demand_csv = """date,sku,quantity
2024-01-01,SKU-001,125
2024-01-02,SKU-001,130
2024-01-03,SKU-001,118
2024-01-04,SKU-001,142
2024-01-05,SKU-001,135
2024-01-06,SKU-001,128
2024-01-07,SKU-001,155
2024-01-08,SKU-001,148
2024-01-09,SKU-001,162
2024-01-10,SKU-001,138
2024-01-11,SKU-001,145
2024-01-12,SKU-001,152
2024-01-13,SKU-001,168
2024-01-14,SKU-001,175
2024-01-15,SKU-001,158
2024-01-16,SKU-001,142
2024-01-17,SKU-001,165
2024-01-18,SKU-001,178
2024-01-19,SKU-001,185
2024-01-20,SKU-001,172
2024-01-21,SKU-001,168
2024-01-22,SKU-001,155
2024-01-23,SKU-001,162
2024-01-24,SKU-001,148
2024-01-25,SKU-001,175
2024-01-26,SKU-001,182
2024-01-27,SKU-001,195
2024-01-28,SKU-001,188
2024-01-29,SKU-001,192
2024-01-30,SKU-001,185"""

# Generate forecast
demand_service = DemandService()
forecast = demand_service.generate_forecast_csv(demand_csv, forecast_horizon_weeks=4)

print("📈 Demand Forecast Generated!")
print(f"Next 4 weeks forecasted demand: {forecast.forecast_summary['total_forecasted_demand']}")
```

### 2. Inventory Optimization (1 minute)

```python
from inventory.service import InventoryService

# Current inventory status
inventory_csv = """sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev
SKU-001,250,14,0.95,100.5,25.2
SKU-002,150,7,0.98,75.0,18.5
SKU-003,300,21,0.90,120.0,30.0"""

# Calculate safety stock
inventory_service = InventoryService()
recommendations = inventory_service.calculate_safety_stock_csv(inventory_csv)

print("📦 Inventory Recommendations:")
for rec in recommendations.recommendations:
    print(f"  {rec.sku}: Safety Stock = {rec.safety_stock}, Action = {rec.recommended_action}")
```

### 3. Route Optimization (2 minutes)

```python
from distribution.service import DistributionService

# Orders to deliver
orders_csv = """order_id,customer_lat,customer_lon,volume_m3,weight_kg
ORD-001,40.7128,-74.0060,2.5,50.0
ORD-002,40.7589,-73.9851,1.8,35.0
ORD-003,40.6892,-74.0445,3.2,65.0
ORD-004,40.7505,-73.9934,2.1,45.0"""

# Available vehicles
vehicles_csv = """vehicle_id,max_volume_m3,max_weight_kg,cost_per_km
VEH-001,25.0,1000.0,1.50
VEH-002,30.0,1200.0,1.75"""

# Optimize routes
distribution_service = DistributionService()
routes = distribution_service.optimize_routes_csv(orders_csv, vehicles_csv)

print("🚛 Route Optimization Complete!")
print(f"Total Distance: {routes.solution.total_distance:.1f} km")
print(f"Total Cost: ${routes.solution.total_cost:.2f}")
```

## Working with DataFrames

All modules support pandas DataFrames for seamless data analysis:

```python
import pandas as pd
from supply.service import SupplyService

# Load data from DataFrame
df = pd.read_csv('your_data.csv')

# Process with any service
supply_service = SupplyService()
result_df = supply_service.optimize_procurement_dataframe(df)

# Continue analysis
result_df['cost_per_unit'] = result_df['total_cost'] / result_df['recommended_quantity']
print(result_df.describe())
```

## API Integration

Quick API setup for web applications:

```python
import requests

# API endpoint
url = "https://api.balancer.platform/v1/demand/forecast"

# Request data
payload = {
    "csv_data": demand_csv,
    "forecast_horizon_weeks": 12,
    "confidence_level": 0.95
}

headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

# Make request
response = requests.post(url, json=payload, headers=headers)
forecast_data = response.json()

print(f"Forecast generated: {forecast_data['forecasts_count']} predictions")
```

## Common Workflows

### End-to-End Supply Chain Optimization

```python
from demand.service import DemandService
from inventory.service import InventoryService
from supply.service import SupplyService

# 1. Generate demand forecast
demand_service = DemandService()
forecast = demand_service.generate_forecast_csv(historical_demand_csv)

# 2. Calculate optimal inventory levels
inventory_service = InventoryService()
inventory_recs = inventory_service.calculate_safety_stock_csv(inventory_csv)

# 3. Optimize procurement
supply_service = SupplyService()
procurement_plan = supply_service.optimize_procurement_csv(
    procurement_csv,
    demand_forecast=forecast.csv_output
)

print("🎯 Complete supply chain optimization finished!")
```

### Real-time Monitoring Dashboard

```python
import time
from datetime import datetime

def monitor_supply_chain():
    while True:
        # Check inventory levels
        current_inventory = get_current_inventory()  # Your data source
        alerts = inventory_service.generate_stock_alerts_csv(current_inventory)
        
        # Check demand patterns
        recent_demand = get_recent_demand()  # Your data source
        demand_analysis = demand_service.analyze_demand_patterns_csv(recent_demand)
        
        # Update dashboard
        update_dashboard({
            'inventory_alerts': alerts,
            'demand_trends': demand_analysis,
            'timestamp': datetime.now()
        })
        
        time.sleep(300)  # Check every 5 minutes

# Start monitoring
monitor_supply_chain()
```

## Next Steps

1. **Explore Module Documentation**: Check out detailed docs for each module:
   - [Supply Module](supply.md)
   - [Inventory Module](inventory.md)
   - [Demand Module](demand.md)
   - [Distribution Module](distribution.md)

2. **Try Advanced Features**:
   - Multi-SKU forecasting
   - Dynamic safety stock calculation
   - Multi-depot route optimization
   - Real-time tracking integration

3. **Integration Examples**:
   - Connect to your ERP system
   - Set up automated workflows
   - Build custom dashboards
   - Create alert systems

4. **Performance Optimization**:
   - Batch processing for large datasets
   - Parallel processing configuration
   - Caching strategies
   - API rate limit management

## Troubleshooting

### Common Issues

**"Insufficient data for forecasting"**
```python
# Ensure minimum 28 data points
if len(demand_data) < 28:
    print("Need at least 28 data points for reliable forecasting")
```

**"Invalid CSV format"**
```python
# Check required columns
required_columns = ['date', 'sku', 'quantity']
if not all(col in df.columns for col in required_columns):
    print(f"Missing columns: {set(required_columns) - set(df.columns)}")
```

**"Optimization failed"**
```python
# Try with relaxed constraints
try:
    result = service.optimize_csv(data)
except OptimizationFailedException:
    result = service.optimize_csv(data, allow_partial_solutions=True)
```

### Getting Help

- 📖 **Documentation**: Check module-specific docs
- 🧪 **Examples**: Look at example files in each domain folder
- 🔍 **Tests**: Review test files for usage patterns
- 💬 **Support**: Contact support@balancer.platform

## Sample Data

Download sample datasets to get started quickly:

```python
from utils.dummy_data import DummyDataGenerator

# Generate sample data
generator = DummyDataGenerator()

# Save sample files
generator.save_sample_data("sample_data/")

print("Sample data generated in sample_data/ folder")
```

This creates:
- `procurement_data.csv` - Sample procurement data
- `manufacturing_data.csv` - Sample manufacturing data
- `inventory_status.csv` - Sample inventory data
- `demand_data.csv` - Sample demand time series
- `orders_and_vehicles.csv` - Sample distribution data

Start with these files to explore the platform capabilities!