# Inventory Management Module

The Inventory Management module provides safety stock calculations, reorder point optimization, and inventory recommendations with comprehensive analytics.

## Overview

The Inventory module optimizes inventory operations through:

- **Safety Stock Calculation**: Statistical methods for optimal safety stock levels
- **Reorder Point Optimization**: Intelligent reorder point recommendations
- **ABC Classification**: Inventory prioritization analysis
- **Stock Alerts**: Real-time inventory monitoring and alerts
- **Flexible Data Handling**: Support for DataFrames, CSV, and JSON formats

## Key Features

### Safety Stock Calculation
- Statistical safety stock calculation
- Service level optimization
- Lead time variability handling
- Demand uncertainty analysis

### Inventory Analytics
- ABC classification analysis
- Turnover metrics calculation
- Stock alert generation
- Performance monitoring

## Data Models

### Core Input Models

#### InventoryStatusInput
```python
class InventoryStatusInput(BaseModel):
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    current_stock: int = Field(..., ge=0, description="Current stock level")
    lead_time_days: int = Field(..., gt=0, description="Lead time in days")
    service_level_target: float = Field(..., gt=0, le=1, description="Target service level")
    avg_weekly_demand: float = Field(..., gt=0, description="Average weekly demand")
    demand_std_dev: float = Field(..., ge=0, description="Demand standard deviation")
```

## Service Layer

### InventoryService Class

The `InventoryService` class provides the main interface for inventory management operations with DataFrame-first architecture.

#### Key Methods

##### calculate_safety_stock(data, params) -> InventoryResult
Calculate optimal safety stock levels and reorder points.

**Parameters:**
- `data`: Input data as DataFrame, CSV string, or list of dictionaries
- `params`: Optional calculation parameters

**Returns:**
- `InventoryResult`: Complete analysis results

**Example Usage:**
```python
from inventory.service import InventoryService
import pandas as pd

service = InventoryService()

# Using DataFrame (recommended)
inventory_data = pd.DataFrame([{
    'sku': 'WIDGET-001',
    'current_stock': 150,
    'lead_time_days': 7,
    'service_level_target': 0.95,
    'avg_weekly_demand': 50.0,
    'demand_std_dev': 12.5
}])

result = service.calculate_safety_stock(inventory_data)
print(f"Analyzed {result.recommendations_count} items")

# Get results in different formats
df_output = service.get_recommendations_dataframe(result)
csv_output = service.get_recommendations_csv(result)
json_output = service.get_recommendations_json(result)
```

##### analyze_abc_classification(data, params) -> ABCResult
Perform ABC analysis for inventory prioritization.

**Example Usage:**
```python
# ABC classification
abc_data = pd.DataFrame([
    {'sku': 'WIDGET-001', 'annual_usage_value': 50000},
    {'sku': 'GADGET-002', 'annual_usage_value': 30000},
    {'sku': 'TOOL-003', 'annual_usage_value': 10000}
])

result = service.analyze_abc_classification(abc_data)
classification_df = service.get_classification_dataframe(result)
```

##### generate_stock_alerts(data, params) -> AlertResult
Generate stock level alerts and recommendations.

**Example Usage:**
```python
# Stock alerts
alert_data = pd.DataFrame([
    {'sku': 'WIDGET-001', 'current_stock': 25, 'reorder_point': 50, 'safety_stock': 30}
])

result = service.generate_stock_alerts(alert_data)
alerts_df = service.get_alerts_dataframe(result)
```

## API Endpoints

### POST `/inventory/optimize`

Calculate safety stock levels and reorder points.

**Request:**
```json
{
  "csv_data": "sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev\nWIDGET-001,150,7,0.95,50.0,12.5",
  "optimization_params": {
    "method": "statistical"
  }
}
```

**Response:**
```json
{
  "success": true,
  "recommendations_count": 1,
  "processing_time_seconds": 0.089,
  "optimization_summary": {
    "total_items_analyzed": 1,
    "items_needing_reorder": 0,
    "avg_service_level": 0.95
  }
}
```

### POST `/inventory/upload`

File upload for inventory analysis.

### GET `/inventory/template`

Download CSV template for inventory data.

## Configuration

### Calculation Parameters

#### Safety Stock Parameters
```python
safety_stock_params = {
    "method": "statistical",           # Calculation method
    "confidence_level": 0.95,          # Statistical confidence level
    "demand_forecast_accuracy": 0.85,  # Forecast accuracy factor
    "supply_variability": 0.10         # Supply variability factor
}
```

#### ABC Classification Parameters
```python
abc_params = {
    "a_threshold": 0.8,    # Top 80% of value = Class A
    "b_threshold": 0.95    # Next 15% of value = Class B
}
```

## Data Formats

### CSV Input Formats

#### Inventory Status Data
```csv
sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev
WIDGET-001,150,7,0.95,50.0,12.5
GADGET-002,75,14,0.90,25.0,8.0
```

#### ABC Classification Data
```csv
sku,annual_usage_value,current_stock,unit_cost
WIDGET-001,50000,150,25.00
GADGET-002,30000,75,40.00
```

## Usage Examples

### Basic Safety Stock Calculation

```python
from inventory.service import InventoryService
import pandas as pd

def calculate_safety_stock():
    service = InventoryService()
    
    # Create DataFrame
    data = pd.DataFrame([
        {
            'sku': 'WIDGET-001',
            'current_stock': 150,
            'lead_time_days': 7,
            'service_level_target': 0.95,
            'avg_weekly_demand': 50.0,
            'demand_std_dev': 12.5
        }
    ])
    
    # Calculation parameters
    params = {
        "method": "statistical",
        "confidence_level": 0.95
    }
    
    # Process calculation
    result = service.calculate_safety_stock(data, params)
    
    print(f"Analyzed {result.recommendations_count} items")
    
    # Get results as DataFrame
    recommendations_df = service.get_recommendations_dataframe(result)
    print(recommendations_df[['sku', 'safety_stock', 'reorder_point', 'recommended_action']])

calculate_safety_stock()
```

### ABC Classification Analysis

```python
def analyze_abc():
    service = InventoryService()
    
    # ABC analysis data
    data = pd.DataFrame([
        {'sku': 'WIDGET-001', 'annual_usage_value': 50000},
        {'sku': 'GADGET-002', 'annual_usage_value': 30000},
        {'sku': 'TOOL-003', 'annual_usage_value': 10000},
        {'sku': 'PART-004', 'annual_usage_value': 5000}
    ])
    
    # Classification parameters
    params = {
        "a_threshold": 0.8,
        "b_threshold": 0.95
    }
    
    # Perform ABC analysis
    result = service.analyze_abc_classification(data, params)
    
    print(f"Classified {len(data)} items")
    
    # Get classification results
    classification_df = service.get_classification_dataframe(result)
    print(classification_df[['sku', 'abc_class', 'annual_usage_value']])

analyze_abc()
```

### Stock Alert Generation

```python
def generate_alerts():
    service = InventoryService()
    
    # Stock data
    data = pd.DataFrame([
        {'sku': 'WIDGET-001', 'current_stock': 25, 'reorder_point': 50, 'safety_stock': 30},
        {'sku': 'GADGET-002', 'current_stock': 100, 'reorder_point': 75, 'safety_stock': 40}
    ])
    
    # Alert parameters
    params = {
        "urgent_threshold": 0.5,
        "warning_threshold": 1.0
    }
    
    # Generate alerts
    result = service.generate_stock_alerts(data, params)
    
    print(f"Generated {result.alert_count} alerts")
    
    # Get alerts
    alerts_df = service.get_alerts_dataframe(result)
    print(alerts_df[['sku', 'alert_type', 'priority', 'message']])

generate_alerts()
```

## Advanced Features

### Turnover Analysis

```python
def analyze_turnover():
    service = InventoryService()
    
    # Turnover data
    data = pd.DataFrame([
        {'sku': 'WIDGET-001', 'annual_demand': 2600, 'avg_inventory_value': 3750},
        {'sku': 'GADGET-002', 'annual_demand': 1300, 'avg_inventory_value': 3000}
    ])
    
    # Calculate turnover metrics
    result = service.calculate_turnover_metrics(data)
    
    turnover_df = service.get_turnover_dataframe(result)
    print(turnover_df[['sku', 'turnover_ratio', 'days_of_inventory', 'turnover_class']])

analyze_turnover()
```

## Error Handling

### Common Exceptions

```python
from inventory.service import InventoryService
import pandas as pd

try:
    service = InventoryService()
    result = service.calculate_safety_stock(data)
except ValueError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Calculation failed: {e}")
```

## Best Practices

### Data Quality
1. **Validate Input Data**: Ensure data completeness and accuracy
2. **Historical Data**: Use sufficient historical data for reliable calculations
3. **Data Consistency**: Maintain consistent units and time periods

### Performance
1. **Use DataFrames**: Prefer DataFrame input for better performance
2. **Batch Processing**: Process multiple SKUs together
3. **Regular Updates**: Update calculations regularly as demand patterns change

### Integration
1. **Multiple Formats**: Support DataFrame, CSV, and JSON as needed
2. **Real-time Monitoring**: Implement continuous stock monitoring
3. **Alert Management**: Set up automated alert systems

---

For more details, see the [API Reference](api.md) and [examples](../inventory/example.py).