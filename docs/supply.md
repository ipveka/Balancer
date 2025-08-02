# Supply Management Module

The Supply Management module provides procurement and manufacturing optimization capabilities with intelligent algorithms for cost optimization and supplier management.

## Overview

The Supply module optimizes supply chain operations through:

- **Procurement Optimization**: Purchase order recommendations with supplier analysis
- **Manufacturing Planning**: Production batch optimization and scheduling
- **Flexible Data Handling**: Support for DataFrames, CSV, and JSON formats
- **Cost Optimization**: Advanced algorithms for minimizing total supply costs

## Key Features

### Procurement Mode
- Purchase order quantity optimization
- Supplier selection and comparison
- Lead time and cost analysis
- Safety stock integration

### Manufacturing Mode
- Production batch size optimization
- Manufacturing schedule planning
- Capacity utilization analysis
- Setup cost optimization

## Data Models

### Core Input Models

#### ProcurementDataInput
```python
class ProcurementDataInput(BaseModel):
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    current_inventory: int = Field(..., ge=0, description="Current inventory level")
    forecast_demand_4weeks: float = Field(..., gt=0, description="4-week demand forecast")
    safety_stock: int = Field(..., ge=0, description="Required safety stock level")
    min_order_qty: int = Field(..., gt=0, description="Minimum order quantity")
    supplier_id: str = Field(..., description="Supplier identifier")
    unit_cost: float = Field(..., gt=0, description="Cost per unit")
```

#### ManufacturingDataInput
```python
class ManufacturingDataInput(BaseModel):
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    current_inventory: int = Field(..., ge=0, description="Current inventory level")
    forecast_demand_4weeks: float = Field(..., gt=0, description="4-week demand forecast")
    safety_stock: int = Field(..., ge=0, description="Required safety stock level")
    batch_size: int = Field(..., gt=0, description="Manufacturing batch size")
    production_time_days: int = Field(..., gt=0, description="Production time in days")
    unit_cost: float = Field(..., gt=0, description="Production cost per unit")
```

## Service Layer

### SupplyService Class

The `SupplyService` class provides the main interface for supply optimization operations with DataFrame-first architecture.

#### Key Methods

##### optimize_procurement(data, params) -> ProcurementResult
Process procurement data and generate optimization recommendations.

**Parameters:**
- `data`: Input data as DataFrame, CSV string, or list of dictionaries
- `params`: Optional optimization parameters

**Returns:**
- `ProcurementResult`: Complete optimization results

**Example Usage:**
```python
from supply.service import SupplyService
import pandas as pd

service = SupplyService()

# Using DataFrame (recommended)
procurement_data = pd.DataFrame([{
    'sku': 'WIDGET-001',
    'current_inventory': 100,
    'forecast_demand_4weeks': 500,
    'safety_stock': 50,
    'min_order_qty': 100,
    'supplier_id': 'SUP-001',
    'unit_cost': 10.50
}])

result = service.optimize_procurement(procurement_data)
print(f"Generated {result.recommendations_count} recommendations")

# Get results in different formats
df_output = service.get_recommendations_dataframe(result)
csv_output = service.get_recommendations_csv(result)
json_output = service.get_recommendations_json(result)
```

##### optimize_manufacturing(data, params) -> ManufacturingResult
Process manufacturing data and generate production recommendations.

**Example Usage:**
```python
# Manufacturing optimization
manufacturing_data = pd.DataFrame([{
    'sku': 'WIDGET-001',
    'current_inventory': 100,
    'forecast_demand_4weeks': 500,
    'safety_stock': 50,
    'batch_size': 200,
    'production_time_days': 5,
    'unit_cost': 8.50
}])

result = service.optimize_manufacturing(manufacturing_data)
recommendations_df = service.get_recommendations_dataframe(result)
```

##### compare_suppliers(data, criteria) -> SupplierComparisonResult
Compare multiple suppliers using multi-criteria analysis.

**Example Usage:**
```python
# Supplier comparison
supplier_data = pd.DataFrame([
    {'sku': 'WIDGET-001', 'supplier_id': 'SUP-A', 'unit_cost': 10.50, 'lead_time_days': 7, 'quality_rating': 4.5},
    {'sku': 'WIDGET-001', 'supplier_id': 'SUP-B', 'unit_cost': 11.25, 'lead_time_days': 5, 'quality_rating': 4.8}
])

criteria = {
    'cost_weight': 0.4,
    'quality_weight': 0.3,
    'lead_time_weight': 0.3
}

result = service.compare_suppliers(supplier_data, criteria)
```

## API Endpoints

### POST `/supply/optimize`

General supply optimization with mode selection.

**Request:**
```json
{
  "mode": "procurement",
  "csv_data": "sku,current_inventory,forecast_demand_4weeks,safety_stock,min_order_qty,supplier_id,unit_cost\nWIDGET-001,100,500,50,100,SUP-001,10.50",
  "optimization_params": {
    "service_level": 0.95
  }
}
```

**Response:**
```json
{
  "success": true,
  "mode": "procurement",
  "recommendations_count": 1,
  "processing_time_seconds": 0.234,
  "optimization_summary": {
    "total_cost": 525.00,
    "total_items": 1,
    "avg_service_level": 0.95
  }
}
```

### POST `/supply/procurement/optimize`

Procurement-specific optimization endpoint.

### POST `/supply/manufacturing/optimize`

Manufacturing-specific optimization endpoint.

## Configuration

### Optimization Parameters

#### Procurement Parameters
```python
procurement_params = {
    "service_level": 0.95,          # Target service level (0.0-1.0)
    "cost_weight": 0.7,             # Cost optimization weight (0.0-1.0)
    "lead_time_buffer": 1.2,        # Lead time safety buffer multiplier
    "demand_variability": 0.15      # Expected demand variability
}
```

#### Manufacturing Parameters
```python
manufacturing_params = {
    "capacity_utilization": 0.85,   # Target capacity utilization (0.0-1.0)
    "setup_cost_weight": 0.3,       # Setup cost optimization weight
    "quality_target": 0.98          # Target quality level
}
```

## Data Formats

### CSV Input Formats

#### Procurement Data
```csv
sku,current_inventory,forecast_demand_4weeks,safety_stock,min_order_qty,supplier_id,unit_cost
WIDGET-001,100,500,50,100,SUP-001,10.50
GADGET-002,75,300,40,75,SUP-002,25.75
```

#### Manufacturing Data
```csv
sku,current_inventory,forecast_demand_4weeks,safety_stock,batch_size,production_time_days,unit_cost
WIDGET-001,100,500,50,200,5,8.50
GADGET-002,25,200,30,100,3,18.75
```

## Usage Examples

### Basic Procurement Optimization

```python
from supply.service import SupplyService
import pandas as pd

def optimize_procurement():
    service = SupplyService()
    
    # Create DataFrame
    data = pd.DataFrame([
        {
            'sku': 'WIDGET-001',
            'current_inventory': 150,
            'forecast_demand_4weeks': 800,
            'safety_stock': 100,
            'min_order_qty': 200,
            'supplier_id': 'SUP-ALPHA',
            'unit_cost': 12.50
        }
    ])
    
    # Optimization parameters
    params = {
        "service_level": 0.95,
        "cost_weight": 0.7
    }
    
    # Process optimization
    result = service.optimize_procurement(data, params)
    
    print(f"Generated {result.recommendations_count} recommendations")
    print(f"Total cost: ${result.optimization_summary['total_cost']:.2f}")
    
    # Get results as DataFrame
    recommendations_df = service.get_recommendations_dataframe(result)
    print(recommendations_df)

optimize_procurement()
```

### Manufacturing Planning

```python
def plan_manufacturing():
    service = SupplyService()
    
    # Manufacturing data
    data = pd.DataFrame([
        {
            'sku': 'WIDGET-001',
            'current_inventory': 100,
            'forecast_demand_4weeks': 500,
            'safety_stock': 50,
            'batch_size': 200,
            'production_time_days': 5,
            'unit_cost': 8.50
        }
    ])
    
    # Manufacturing parameters
    params = {
        "capacity_utilization": 0.85,
        "setup_cost_weight": 0.3
    }
    
    # Process manufacturing optimization
    result = service.optimize_manufacturing(data, params)
    
    print(f"Generated {result.recommendations_count} production recommendations")
    
    # Export as CSV
    csv_output = service.get_recommendations_csv(result)
    print("Manufacturing Schedule:")
    print(csv_output)

plan_manufacturing()
```

## Error Handling

### Common Exceptions

```python
from supply.service import SupplyService
import pandas as pd

try:
    service = SupplyService()
    result = service.optimize_procurement(data)
except ValueError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Optimization failed: {e}")
```

## Best Practices

### Data Quality
1. **Validate Input Data**: Always validate data before processing
2. **Handle Missing Values**: Implement strategies for missing data
3. **Data Consistency**: Ensure consistent units and formats

### Performance
1. **Use DataFrames**: Prefer DataFrame input for better performance
2. **Batch Processing**: Use batch processing for large datasets
3. **Parameter Tuning**: Regularly tune optimization parameters

### Integration
1. **Multiple Formats**: Support DataFrame, CSV, and JSON as needed
2. **Error Handling**: Implement comprehensive error handling
3. **Monitoring**: Monitor optimization performance and results

---

For more details, see the [API Reference](api.md) and [examples](../supply/example.py).