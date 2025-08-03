# API Reference

This document provides the API reference for the Balancer supply chain optimization platform. All endpoints return JSON responses with consistent error handling.

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Common Response Format

```json
{
  "success": true,
  "recommendations_count": 5,
  "processing_time_seconds": 0.123,
  "optimization_summary": {
    "total_cost": 1250.00,
    "total_items": 5
  }
}
```

## Supply Management API

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
  "csv_output": "sku,recommended_quantity,supplier_id,order_date,expected_delivery,total_cost\n...",
  "processing_time_seconds": 0.234,
  "optimization_summary": {
    "total_cost": 525.00,
    "total_items": 1,
    "avg_service_level": 0.95
  }
}
```

### POST `/supply/procurement/optimize`

Procurement-specific optimization.

### POST `/supply/manufacturing/optimize`

Manufacturing-specific optimization.

### POST `/supply/procurement/upload`

File upload for procurement optimization.

### POST `/supply/manufacturing/upload`

File upload for manufacturing optimization.

## Inventory Management API

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
  "csv_output": "sku,safety_stock,reorder_point,current_stock,recommended_action,days_until_stockout,confidence_score\n...",
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

### GET `/inventory/actions`

Get supported recommendation actions.

## Demand Forecasting API

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
  "csv_output": "sku,forecast_date,prediction,confidence_interval_lower,confidence_interval_upper\n...",
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

### GET `/demand/models/{sku}`

Get model information for specific SKU.

## Distribution Planning API

### POST `/distribution/optimize-routes`

Optimize vehicle routes using VRP algorithms.

**Request:**
```json
{
  "request_id": "REQ-001",
  "orders": [
    {
      "order_id": "ORD-001",
      "customer_lat": 40.7831,
      "customer_lon": -73.9712,
      "volume_m3": 2.5,
      "weight_kg": 150
    }
  ],
  "vehicles": [
    {
      "vehicle_id": "VEH-001",
      "max_volume_m3": 20.0,
      "max_weight_kg": 1000,
      "cost_per_km": 0.50
    }
  ],
  "algorithm_preference": "greedy",
  "distribution_center": {
    "center_id": "DC-001",
    "latitude": 40.7589,
    "longitude": -73.9851
  }
}
```

**Response:**
```json
{
  "request_id": "REQ-001",
  "solution": {
    "solution_id": "SOL-001",
    "total_distance": 45.2,
    "total_cost": 22.60,
    "algorithm_used": "greedy",
    "optimization_time_seconds": 0.234,
    "num_routes": 1,
    "num_orders_assigned": 1
  },
  "routes": [
    {
      "route_id": "ROUTE-001",
      "vehicle_id": "VEH-001",
      "order_sequence": ["ORD-001"],
      "total_distance_km": 45.2,
      "total_cost": 22.60,
      "estimated_duration_hours": 6.0
    }
  ],
  "assignments": [
    {
      "vehicle_id": "VEH-001",
      "order_id": "ORD-001",
      "sequence": 1,
      "distance_km": 22.6,
      "total_cost": 11.30
    }
  ],
  "optimization_status": "success"
}
```

### POST `/distribution/optimize-csv`

CSV-based route optimization.

### GET `/distribution/health`

Distribution service health check.

## Utility Endpoints

### GET `/health`

API health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "modules": {
    "supply": "operational",
    "inventory": "operational",
    "demand": "operational",
    "distribution": "operational"
  }
}
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": "Missing required column: sku"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## CSV Data Formats

### Supply Management

**Procurement:**
```csv
sku,current_inventory,forecast_demand_4weeks,safety_stock,min_order_qty,supplier_id,unit_cost
WIDGET-001,100,500,50,100,SUP-001,10.50
```

**Manufacturing:**
```csv
sku,current_inventory,forecast_demand_4weeks,safety_stock,batch_size,production_time_days,unit_cost
WIDGET-001,100,500,50,200,5,8.50
```

### Inventory Management

```csv
sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev
WIDGET-001,150,7,0.95,50.0,12.5
```

### Demand Forecasting

```csv
date,sku,quantity
2024-01-01,WIDGET-001,150
2024-01-02,WIDGET-001,142
```

### Distribution Planning

**Orders:**
```csv
order_id,customer_lat,customer_lon,volume_m3,weight_kg
ORD-001,40.7831,-73.9712,2.5,150
```

**Vehicles:**
```csv
vehicle_id,max_volume_m3,max_weight_kg,cost_per_km
VEH-001,20.0,1000,0.50
```

## Rate Limiting

- **Default**: 100 requests per minute per IP
- **Headers**: 
  - `X-RateLimit-Limit`: Maximum requests per window
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset time

---

For interactive testing, visit http://localhost:8000/docs