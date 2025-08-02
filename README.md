# Balancer: Supply Chain Optimization Platform

A modern supply chain optimization platform that provides intelligent solutions for procurement, inventory management, demand forecasting, and distribution planning. Built with FastAPI and machine learning, it offers both REST APIs and programmatic interfaces.

## 🚀 Key Features

- **Supply Optimization**: Procurement and manufacturing planning with cost optimization
- **Inventory Management**: Safety stock calculations and reorder point optimization  
- **Demand Forecasting**: AI-powered predictions using LightGBM machine learning
- **Distribution Planning**: Vehicle routing optimization and logistics management
- **Flexible Data Handling**: Support for DataFrames, CSV, and JSON formats
- **Production Ready**: FastAPI with automatic documentation and Docker support

## 🛠 Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload

# Access the API
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
```

## 💡 Usage Examples

### Programmatic Usage (Recommended)

```python
from supply.service import SupplyService
from inventory.service import InventoryService
import pandas as pd

# Supply optimization
supply_service = SupplyService()
procurement_data = pd.DataFrame([{
    'sku': 'WIDGET-001',
    'current_inventory': 100,
    'forecast_demand_4weeks': 500,
    'safety_stock': 50,
    'min_order_qty': 100,
    'supplier_id': 'SUP-001',
    'unit_cost': 10.50
}])

result = supply_service.optimize_procurement(procurement_data)
recommendations_df = supply_service.get_recommendations_dataframe(result)
print(f"Generated {result.recommendations_count} recommendations")

# Inventory management
inventory_service = InventoryService()
inventory_data = pd.DataFrame([{
    'sku': 'WIDGET-001',
    'current_stock': 150,
    'lead_time_days': 7,
    'service_level_target': 0.95,
    'avg_weekly_demand': 50.0,
    'demand_std_dev': 12.5
}])

result = inventory_service.calculate_safety_stock(inventory_data)
recommendations_df = inventory_service.get_recommendations_dataframe(result)
print(recommendations_df[['sku', 'safety_stock', 'reorder_point', 'recommended_action']])
```

### API Usage

```python
import requests

# Supply optimization
response = requests.post("http://localhost:8000/supply/optimize", json={
    "mode": "procurement",
    "csv_data": "sku,current_inventory,forecast_demand_4weeks,safety_stock,min_order_qty,supplier_id,unit_cost\nWIDGET-001,100,500,50,100,SUP-001,10.50"
})

result = response.json()
print(f"Success: {result['success']}, Recommendations: {result['recommendations_count']}")
```

## 📚 API Endpoints

### Supply Management (`/supply`)
- `POST /supply/optimize` - General supply optimization
- `POST /supply/procurement/optimize` - Procurement optimization
- `POST /supply/manufacturing/optimize` - Manufacturing optimization

### Inventory Management (`/inventory`)
- `POST /inventory/optimize` - Safety stock and reorder point calculation
- `POST /inventory/upload` - File upload for inventory analysis

### Demand Forecasting (`/demand`)
- `POST /demand/forecast` - AI-powered demand forecasting
- `POST /demand/upload` - File upload for forecasting

### Distribution Planning (`/distribution`)
- `POST /distribution/optimize-routes` - Vehicle routing optimization
- `POST /distribution/optimize-csv` - CSV-based route optimization

## 🔧 Data Formats

The system supports multiple input/output formats:

```python
# DataFrame (recommended for programmatic use)
result = service.optimize_procurement(dataframe)

# CSV string (for API endpoints)
result = service.optimize_procurement(csv_string)

# List of dictionaries (for JSON APIs)
result = service.optimize_procurement(dict_list)

# Multiple output formats
df_output = service.get_recommendations_dataframe(result)
csv_output = service.get_recommendations_csv(result)
json_output = service.get_recommendations_json(result)
```

## 📖 Documentation

- [Supply Management](docs/supply.md) - Procurement and manufacturing
- [Inventory Management](docs/inventory.md) - Safety stock calculations
- [Demand Forecasting](docs/demand.md) - AI-powered predictions
- [Distribution Planning](docs/distribution.md) - Route optimization
- [API Reference](docs/api.md) - Complete API documentation

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Test specific modules
pytest tests/test_supply.py
pytest tests/test_inventory.py
```

## 🚀 Deployment

```bash
# Docker deployment
docker build -t balancer .
docker run -p 8000:8000 balancer

# Production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📋 Requirements

- Python 3.8+
- FastAPI, pandas, numpy, scikit-learn, lightgbm
- See `requirements.txt` for complete list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

**Balancer** - Intelligent supply chain optimization 🚀