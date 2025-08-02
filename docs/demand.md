# Demand Forecasting Module

The Demand Forecasting module provides AI-powered demand prediction using LightGBM machine learning with comprehensive analytics and pattern recognition.

## Overview

The Demand module provides intelligent forecasting through:

- **AI-Powered Forecasting**: LightGBM machine learning for accurate predictions
- **Feature Engineering**: Automated feature creation from time series data
- **Seasonality Detection**: Automatic identification of seasonal patterns
- **Confidence Intervals**: Statistical confidence bounds for predictions
- **Pattern Analysis**: Trend and volatility analysis
- **Flexible Data Handling**: Support for DataFrames, CSV, and JSON formats

## Key Features

### Machine Learning Forecasting
- LightGBM gradient boosting models
- Automated feature engineering
- Lag features and moving averages
- Seasonality components
- Trend analysis

### Analytics and Insights
- Forecast accuracy metrics
- Pattern recognition
- Outlier detection
- Data quality assessment

## Data Models

### Core Input Models

#### DemandDataInput
```python
class DemandDataInput(BaseModel):
    date: date = Field(..., description="Date of demand observation")
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    quantity: float = Field(..., ge=0, description="Demand quantity")
```

## Service Layer

### DemandService Class

The `DemandService` class provides the main interface for demand forecasting operations with DataFrame-first architecture.

#### Key Methods

##### generate_forecast(data, horizon_weeks, params) -> DemandResult
Generate AI-powered demand forecasts using LightGBM.

**Parameters:**
- `data`: Input data as DataFrame, CSV string, or list of dictionaries
- `horizon_weeks`: Number of weeks to forecast ahead
- `params`: Optional model parameters

**Returns:**
- `DemandResult`: Complete forecasting results

**Example Usage:**
```python
from demand.service import DemandService
import pandas as pd

service = DemandService()

# Using DataFrame (recommended)
demand_data = pd.DataFrame([
    {'date': '2024-01-01', 'sku': 'WIDGET-001', 'quantity': 150},
    {'date': '2024-01-02', 'sku': 'WIDGET-001', 'quantity': 142},
    {'date': '2024-01-03', 'sku': 'WIDGET-001', 'quantity': 158},
    # ... more historical data
])

result = service.generate_forecast(demand_data, forecast_horizon_weeks=12)
print(f"Generated {result.forecasts_count} forecasts")

# Get results in different formats
df_output = service.get_forecasts_dataframe(result)
csv_output = service.get_forecasts_csv(result)
json_output = service.get_forecasts_json(result)
```

##### analyze_demand_patterns(data, params) -> AnalyticsResult
Analyze historical demand patterns and trends.

**Example Usage:**
```python
# Pattern analysis
result = service.analyze_demand_patterns(demand_data, {
    "include_seasonality": True,
    "detect_outliers": True
})

analytics_df = service.get_analytics_dataframe(result)
```

## API Endpoints

### POST `/demand/forecast`

Generate AI-powered demand forecasts.

**Request:**
```json
{
  "csv_data": "date,sku,quantity\n2024-01-01,WIDGET-001,150\n2024-01-02,WIDGET-001,142",
  "forecast_horizon_weeks": 12,
  "confidence_level": 0.95,
  "include_seasonality": true
}
```

**Response:**
```json
{
  "success": true,
  "forecasts_count": 12,
  "processing_time_seconds": 1.234,
  "model_performance": {
    "mae": 12.5,
    "mape": 8.2,
    "rmse": 18.7,
    "directional_accuracy": 85.5
  },
  "forecast_summary": {
    "total_forecasted_demand": 1742.4,
    "trend_direction": "INCREASING",
    "seasonality_detected": true,
    "confidence_score": 0.92
  }
}
```

### POST `/demand/upload`

File upload for demand forecasting.

### POST `/demand/analytics`

Analyze demand patterns and trends.

### GET `/demand/template`

Download CSV template for demand data.

## Configuration

### Model Parameters

#### Forecasting Parameters
```python
forecast_params = {
    "confidence_level": 0.95,          # Confidence level for intervals
    "include_seasonality": True,       # Include seasonal components
    "model_type": "lightgbm",         # ML model type
    "feature_engineering": True       # Enable feature engineering
}
```

#### LightGBM Parameters
```python
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9
}
```

## Data Formats

### CSV Input Format

#### Historical Demand Data
```csv
date,sku,quantity
2024-01-01,WIDGET-001,150
2024-01-02,WIDGET-001,142
2024-01-03,WIDGET-001,158
2024-01-04,WIDGET-001,135
```

## Usage Examples

### Basic Demand Forecasting

```python
from demand.service import DemandService
import pandas as pd

def generate_forecasts():
    service = DemandService()
    
    # Create historical demand data
    data = pd.DataFrame([
        {'date': '2024-01-01', 'sku': 'WIDGET-001', 'quantity': 150},
        {'date': '2024-01-02', 'sku': 'WIDGET-001', 'quantity': 142},
        {'date': '2024-01-03', 'sku': 'WIDGET-001', 'quantity': 158},
        {'date': '2024-01-04', 'sku': 'WIDGET-001', 'quantity': 135},
        {'date': '2024-01-05', 'sku': 'WIDGET-001', 'quantity': 167}
        # ... more historical data needed for reliable forecasting
    ])
    
    # Forecasting parameters
    params = {
        "confidence_level": 0.95,
        "include_seasonality": True
    }
    
    # Generate forecasts
    result = service.generate_forecast(data, forecast_horizon_weeks=12, model_params=params)
    
    print(f"Generated {result.forecasts_count} forecasts")
    print(f"Model performance - MAPE: {result.model_performance['mape']:.1f}%")
    
    # Get forecasts as DataFrame
    forecasts_df = service.get_forecasts_dataframe(result)
    print(forecasts_df[['sku', 'forecast_date', 'prediction', 'confidence_interval_lower', 'confidence_interval_upper']])

generate_forecasts()
```

### Demand Pattern Analysis

```python
def analyze_patterns():
    service = DemandService()
    
    # Historical data for analysis
    data = pd.DataFrame([
        # ... historical demand data
    ])
    
    # Analysis parameters
    params = {
        "include_seasonality": True,
        "detect_outliers": True,
        "trend_analysis": True
    }
    
    # Analyze patterns
    result = service.analyze_demand_patterns(data, params)
    
    print(f"Analyzed patterns for {result.skus_analyzed} SKUs")
    
    # Get analytics results
    analytics_df = service.get_analytics_dataframe(result)
    print(analytics_df[['sku', 'trend_direction', 'seasonality_pattern', 'volatility']])

analyze_patterns()
```

### Model Performance Validation

```python
def validate_forecasts():
    service = DemandService()
    
    # Split data into train/test
    train_data = historical_data[:-30]  # All but last 30 days
    test_data = historical_data[-30:]   # Last 30 days
    
    # Generate forecasts on training data
    result = service.generate_forecast(train_data, forecast_horizon_weeks=4)
    
    # Validate against test data
    validation_result = service.validate_forecast_accuracy(result, test_data)
    
    print(f"Validation MAPE: {validation_result['mape']:.1f}%")
    print(f"Directional Accuracy: {validation_result['directional_accuracy']:.1f}%")

validate_forecasts()
```

## Advanced Features

### Feature Engineering

The service automatically creates features from time series data:

- **Lag Features**: Previous demand values (lag 1, 7, 14, 30 days)
- **Moving Averages**: Rolling averages (7, 14, 30 day windows)
- **Seasonality**: Day of week, month, quarter effects
- **Trend Components**: Linear and polynomial trends
- **Statistical Features**: Rolling standard deviation, min/max

### Model Management

```python
# Get model information for a specific SKU
model_info = service.get_model_summary('WIDGET-001')
print(f"Model type: {model_info['model_type']}")
print(f"Training date: {model_info['training_date']}")
print(f"Performance: {model_info['performance_metrics']}")

# Feature importance
importance = model_info['feature_importance']
for feature, score in importance.items():
    print(f"{feature}: {score:.3f}")
```

### Forecast Accuracy Levels

The system classifies forecast accuracy:

- **HIGH**: MAPE < 10% - Very reliable for planning
- **MEDIUM**: MAPE 10-20% - Moderately reliable
- **LOW**: MAPE > 20% - Use with caution

## Error Handling

### Common Exceptions

```python
from demand.service import DemandService
import pandas as pd

try:
    service = DemandService()
    result = service.generate_forecast(data, forecast_horizon_weeks=12)
except ValueError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Forecasting failed: {e}")
```

### Data Requirements

- **Minimum Data**: At least 28 data points (4 weeks) for reliable forecasting
- **Data Quality**: Regular time intervals, minimal missing values
- **Seasonality**: At least 2 full seasonal cycles for seasonal patterns

## Best Practices

### Data Quality
1. **Sufficient History**: Provide at least 3-6 months of historical data
2. **Regular Intervals**: Ensure consistent time intervals in data
3. **Handle Outliers**: Clean or flag unusual demand spikes
4. **Missing Values**: Handle gaps in historical data appropriately

### Model Performance
1. **Regular Retraining**: Retrain models monthly or quarterly
2. **Validation**: Regularly validate forecasts against actual demand
3. **Parameter Tuning**: Adjust model parameters based on performance
4. **Multiple Models**: Consider ensemble approaches for critical SKUs

### Integration
1. **Multiple Formats**: Support DataFrame, CSV, and JSON as needed
2. **Real-time Updates**: Update forecasts as new data becomes available
3. **Confidence Intervals**: Always consider forecast uncertainty
4. **Business Context**: Incorporate business knowledge and constraints

---

For more details, see the [API Reference](api.md) and [examples](../demand/example.py).