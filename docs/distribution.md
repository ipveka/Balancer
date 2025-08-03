# Distribution Planning Module

The Distribution Planning module provides vehicle routing optimization and logistics management with advanced VRP algorithms and capacity constraint handling.

## Overview

The Distribution module optimizes logistics operations through:

- **Route Optimization**: Vehicle Routing Problem (VRP) solving with multiple algorithms
- **Capacity Management**: Vehicle capacity and constraint optimization
- **Distance Calculation**: Haversine and Euclidean distance calculations
- **Distribution Centers**: Multi-depot routing and management
- **Performance Analytics**: Route performance monitoring and optimization
- **Flexible Data Handling**: Support for DataFrames, CSV, and JSON formats

## Key Features

### Route Optimization
- Greedy and nearest neighbor algorithms
- Capacity constraint handling
- Distance matrix optimization
- Multi-vehicle assignment

### Logistics Management
- Distribution center management
- Vehicle tracking and monitoring
- Route performance analytics
- Cost optimization

## Data Models

### Core Input Models

#### OrderInput
```python
class OrderInput(BaseModel):
    order_id: str = Field(..., description="Unique order identifier")
    customer_lat: float = Field(..., description="Customer latitude")
    customer_lon: float = Field(..., description="Customer longitude")
    volume_m3: float = Field(..., ge=0, description="Order volume in cubic meters")
    weight_kg: float = Field(..., ge=0, description="Order weight in kilograms")
```

#### VehicleInput
```python
class VehicleInput(BaseModel):
    vehicle_id: str = Field(..., description="Unique vehicle identifier")
    max_volume_m3: float = Field(..., gt=0, description="Maximum volume capacity")
    max_weight_kg: float = Field(..., gt=0, description="Maximum weight capacity")
    cost_per_km: float = Field(..., ge=0, description="Cost per kilometer")
```

## Service Layer

### DistributionService Class

The `DistributionService` class provides the main interface for distribution optimization operations with DataFrame-first architecture.

#### Key Methods

##### optimize_routes(orders_data, vehicles_data, params) -> DistributionResult
Optimize vehicle routes using VRP algorithms.

**Parameters:**
- `orders_data`: Orders data as DataFrame, CSV string, or list of dictionaries
- `vehicles_data`: Vehicles data as DataFrame, CSV string, or list of dictionaries
- `params`: Optional optimization parameters

**Returns:**
- `DistributionResult`: Complete optimization results

**Example Usage:**
```python
from distribution.service import DistributionService
import pandas as pd

service = DistributionService()

# Using DataFrames (recommended)
orders_data = pd.DataFrame([
    {
        'order_id': 'ORD-001',
        'customer_lat': 40.7831,
        'customer_lon': -73.9712,
        'volume_m3': 2.5,
        'weight_kg': 150
    }
])

vehicles_data = pd.DataFrame([
    {
        'vehicle_id': 'VEH-001',
        'max_volume_m3': 20.0,
        'max_weight_kg': 1000,
        'cost_per_km': 0.50
    }
])

optimization_params = {
    "algorithm_preference": "greedy",
    "depot_lat": 40.7589,
    "depot_lon": -73.9851
}

result = service.optimize_routes(orders_data, vehicles_data, optimization_params)
print(f"Optimized {len(result.assignments)} order assignments")

# Get results in different formats
assignments_df = service.get_assignments_dataframe(result)
csv_output = service.get_assignments_csv(result)
json_output = service.get_assignments_json(result)
```

##### calculate_distance_matrix(locations) -> DistanceMatrix
Calculate distance matrix between locations.

**Example Usage:**
```python
# Distance matrix calculation
locations = pd.DataFrame([
    {'id': 'DEPOT', 'lat': 40.7589, 'lon': -73.9851},
    {'id': 'CUST-001', 'lat': 40.7831, 'lon': -73.9712},
    {'id': 'CUST-002', 'lat': 40.7505, 'lon': -73.9934}
])

distance_matrix = service.calculate_distance_matrix(locations)
```

##### analyze_route_performance(routes_data) -> PerformanceResult
Analyze route performance and efficiency.

**Example Usage:**
```python
# Route performance analysis
performance_result = service.analyze_route_performance(routes_data)
performance_df = service.get_performance_dataframe(performance_result)
```

## API Endpoints

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

## Configuration

### Optimization Parameters

#### Route Optimization Parameters
```python
optimization_params = {
    "algorithm_preference": "greedy",      # Algorithm choice
    "depot_lat": 40.7589,                 # Depot latitude
    "depot_lon": -73.9851,                # Depot longitude
    "max_route_distance_km": 100.0,       # Maximum route distance
    "optimization_timeout_seconds": 300    # Optimization timeout
}
```

#### Algorithm Options
- **greedy**: Fast greedy assignment algorithm
- **nearest_neighbor**: Nearest neighbor heuristic
- **savings**: Clarke-Wright savings algorithm (future)
- **genetic**: Genetic algorithm optimization (future)

## Data Formats

### CSV Input Formats

#### Orders Data
```csv
order_id,customer_lat,customer_lon,volume_m3,weight_kg
ORD-001,40.7831,-73.9712,2.5,150
ORD-002,40.7505,-73.9934,1.8,120
```

#### Vehicles Data
```csv
vehicle_id,max_volume_m3,max_weight_kg,cost_per_km
VEH-001,20.0,1000,0.50
VEH-002,15.0,800,0.45
```

## Usage Examples

### Basic Route Optimization

```python
from distribution.service import DistributionService
import pandas as pd

def optimize_routes():
    service = DistributionService()
    
    # Create orders DataFrame
    orders = pd.DataFrame([
        {
            'order_id': 'ORD-001',
            'customer_lat': 40.7831,
            'customer_lon': -73.9712,
            'volume_m3': 2.5,
            'weight_kg': 150
        },
        {
            'order_id': 'ORD-002',
            'customer_lat': 40.7505,
            'customer_lon': -73.9934,
            'volume_m3': 1.8,
            'weight_kg': 120
        }
    ])
    
    # Create vehicles DataFrame
    vehicles = pd.DataFrame([
        {
            'vehicle_id': 'VEH-001',
            'max_volume_m3': 20.0,
            'max_weight_kg': 1000,
            'cost_per_km': 0.50
        }
    ])
    
    # Optimization parameters
    params = {
        "algorithm_preference": "greedy",
        "depot_lat": 40.7589,
        "depot_lon": -73.9851
    }
    
    # Optimize routes
    result = service.optimize_routes(orders, vehicles, params)
    
    print(f"Solution ID: {result.solution['solution_id']}")
    print(f"Total distance: {result.solution['total_distance']:.1f} km")
    print(f"Total cost: ${result.solution['total_cost']:.2f}")
    
    # Get assignments as DataFrame
    assignments_df = service.get_assignments_dataframe(result)
    print(assignments_df[['vehicle_id', 'order_id', 'sequence', 'distance_km']])

optimize_routes()
```

### Distance Matrix Calculation

```python
def calculate_distances():
    service = DistributionService()
    
    # Define locations
    locations = pd.DataFrame([
        {'id': 'DEPOT', 'lat': 40.7589, 'lon': -73.9851},
        {'id': 'CUST-001', 'lat': 40.7831, 'lon': -73.9712},
        {'id': 'CUST-002', 'lat': 40.7505, 'lon': -73.9934},
        {'id': 'CUST-003', 'lat': 40.7282, 'lon': -74.0776}
    ])
    
    # Calculate distance matrix
    distance_matrix = service.calculate_distance_matrix(locations)
    
    print("Distance Matrix (km):")
    print(distance_matrix)

calculate_distances()
```

### Route Performance Analysis

```python
def analyze_performance():
    service = DistributionService()
    
    # Route performance data
    routes_data = pd.DataFrame([
        {
            'route_id': 'ROUTE-001',
            'vehicle_id': 'VEH-001',
            'total_distance_km': 45.2,
            'total_cost': 22.60,
            'orders_delivered': 3,
            'delivery_time_hours': 6.5
        }
    ])
    
    # Analyze performance
    result = service.analyze_route_performance(routes_data)
    
    performance_df = service.get_performance_dataframe(result)
    print(performance_df[['route_id', 'efficiency_score', 'cost_per_delivery', 'utilization_rate']])

analyze_performance()
```

## Advanced Features

### Multi-Depot Routing

```python
# Multiple distribution centers
distribution_centers = pd.DataFrame([
    {'center_id': 'DC-NYC', 'lat': 40.7589, 'lon': -73.9851},
    {'center_id': 'DC-BOS', 'lat': 42.3601, 'lon': -71.0589}
])

# Optimize with multiple depots
result = service.optimize_multi_depot_routes(orders, vehicles, distribution_centers)
```

### Capacity Utilization Analysis

```python
def analyze_capacity():
    service = DistributionService()
    
    # Analyze vehicle capacity utilization
    utilization_result = service.analyze_capacity_utilization(assignments, orders, vehicles)
    
    utilization_df = service.get_utilization_dataframe(utilization_result)
    print(utilization_df[['vehicle_id', 'volume_utilization', 'weight_utilization']])

analyze_capacity()
```

### Time Window Constraints

```python
# Orders with delivery time windows
orders_with_windows = pd.DataFrame([
    {
        'order_id': 'ORD-001',
        'customer_lat': 40.7831,
        'customer_lon': -73.9712,
        'volume_m3': 2.5,
        'weight_kg': 150,
        'delivery_window_start': '09:00',
        'delivery_window_end': '17:00'
    }
])

# Optimize with time constraints
result = service.optimize_routes_with_time_windows(orders_with_windows, vehicles)
```

## Error Handling

### Common Exceptions

```python
from distribution.service import DistributionService
import pandas as pd

try:
    service = DistributionService()
    result = service.optimize_routes(orders_data, vehicles_data)
except ValueError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Route optimization failed: {e}")
```

### Capacity Validation

```python
# Validate vehicle capacity constraints
try:
    result = service.optimize_routes(orders, vehicles)
    
    # Check for capacity violations
    violations = service.validate_capacity_constraints(result.assignments, orders, vehicles)
    if violations:
        print(f"Capacity violations detected: {violations}")
        
except Exception as e:
    print(f"Capacity validation failed: {e}")
```

## Best Practices

### Data Quality
1. **Accurate Coordinates**: Ensure precise latitude/longitude coordinates
2. **Realistic Constraints**: Set appropriate vehicle capacity limits
3. **Complete Data**: Provide all required order and vehicle information

### Performance
1. **Algorithm Selection**: Choose appropriate algorithm for problem size
2. **Batch Processing**: Process multiple routes together for efficiency
3. **Distance Caching**: Cache distance calculations for repeated use

### Integration
1. **Multiple Formats**: Support DataFrame, CSV, and JSON as needed
2. **Real-time Updates**: Update routes as new orders arrive
3. **Monitoring**: Track route performance and optimization metrics
4. **Scalability**: Design for handling large numbers of orders and vehicles

### Optimization Strategy
1. **Problem Size**: Use greedy for large problems, more sophisticated algorithms for smaller ones
2. **Constraints**: Balance between optimization quality and computation time
3. **Validation**: Always validate solutions against business constraints
4. **Continuous Improvement**: Monitor and refine optimization parameters

---

For more details, see the [API Reference](api.md) and [examples](../distribution/example.py).