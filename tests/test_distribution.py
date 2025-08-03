"""
Tests for the distribution module.

This module contains comprehensive tests for distribution functionality,
including API endpoints, service logic, route optimization algorithms,
and capacity constraint validation.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import io
from datetime import datetime

from distribution.api import router
from distribution.models import (
    OrderInput, VehicleInput, OrdersAndVehiclesCSV, RouteAssignment,
    DistributionCenter, RouteOptimizationRequest, GeographicalLocation,
    VehicleCapacityUtilization
)
from distribution.service import DistributionService


# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestDistributionModels:
    """Test distribution data models and validation."""
    
    def test_order_input_validation(self):
        """Test OrderInput model validation."""
        # Valid order
        order = OrderInput(
            order_id="ORD001",
            customer_lat=40.7128,
            customer_lon=-74.0060,
            volume_m3=2.5,
            weight_kg=150.0
        )
        assert order.order_id == "ORD001"
        assert order.customer_lat == 40.7128
        assert order.volume_m3 == 2.5
        
        # Invalid latitude
        with pytest.raises(ValueError):
            OrderInput(
                order_id="ORD002",
                customer_lat=95.0,  # Invalid latitude
                customer_lon=-74.0060,
                volume_m3=2.5,
                weight_kg=150.0
            )
        
        # Invalid longitude
        with pytest.raises(ValueError):
            OrderInput(
                order_id="ORD003",
                customer_lat=40.7128,
                customer_lon=185.0,  # Invalid longitude
                volume_m3=2.5,
                weight_kg=150.0
            )
        
        # Invalid volume
        with pytest.raises(ValueError):
            OrderInput(
                order_id="ORD004",
                customer_lat=40.7128,
                customer_lon=-74.0060,
                volume_m3=-1.0,  # Invalid volume
                weight_kg=150.0
            )
    
    def test_vehicle_input_validation(self):
        """Test VehicleInput model validation."""
        # Valid vehicle
        vehicle = VehicleInput(
            vehicle_id="VEH001",
            max_volume_m3=10.0,
            max_weight_kg=500.0,
            cost_per_km=2.5
        )
        assert vehicle.vehicle_id == "VEH001"
        assert vehicle.max_volume_m3 == 10.0
        
        # Invalid capacity values
        with pytest.raises(ValueError):
            VehicleInput(
                vehicle_id="VEH002",
                max_volume_m3=0.0,  # Invalid volume
                max_weight_kg=500.0,
                cost_per_km=2.5
            )
    
    def test_route_assignment_validation(self):
        """Test RouteAssignment model validation."""
        # Valid assignment
        assignment = RouteAssignment(
            vehicle_id="VEH001",
            order_id="ORD001",
            sequence=1,
            distance_km=5.2,
            total_cost=13.0
        )
        assert assignment.sequence == 1
        assert assignment.distance_km == 5.2
        
        # Invalid sequence
        with pytest.raises(ValueError):
            RouteAssignment(
                vehicle_id="VEH001",
                order_id="ORD001",
                sequence=0,  # Invalid sequence
                distance_km=5.2,
                total_cost=13.0
            )
    
    def test_distribution_center_validation(self):
        """Test DistributionCenter model validation."""
        # Valid distribution center
        center = DistributionCenter(
            center_id="DC001",
            name="Test Center",
            latitude=40.7831,
            longitude=-73.9712,
            max_vehicles=20,
            operating_hours_start="06:00",
            operating_hours_end="22:00"
        )
        assert center.center_id == "DC001"
        assert center.max_vehicles == 20
        
        # Invalid time format
        with pytest.raises(ValueError):
            DistributionCenter(
                center_id="DC002",
                name="Test Center 2",
                latitude=40.7831,
                longitude=-73.9712,
                max_vehicles=20,
                operating_hours_start="25:00",  # Invalid hour
                operating_hours_end="22:00"
            )
    
    def test_geographical_location_distance_calculation(self):
        """Test geographical distance calculations."""
        loc1 = GeographicalLocation(latitude=40.7128, longitude=-74.0060)
        loc2 = GeographicalLocation(latitude=40.7589, longitude=-73.9851)
        
        # Test haversine distance
        haversine_dist = loc1.distance_to(loc2, method="haversine")
        assert isinstance(haversine_dist, float)
        assert haversine_dist > 0
        
        # Test euclidean distance
        euclidean_dist = loc1.distance_to(loc2, method="euclidean")
        assert isinstance(euclidean_dist, float)
        assert euclidean_dist > 0
        
        # Test invalid method
        with pytest.raises(ValueError):
            loc1.distance_to(loc2, method="invalid")


class TestDistributionService:
    """Test distribution service functionality."""
    
    @pytest.fixture
    def service(self):
        """Create distribution service instance."""
        return DistributionService()
    
    @pytest.fixture
    def sample_orders(self):
        """Create sample orders for testing."""
        return [
            OrderInput(
                order_id="ORD001",
                customer_lat=40.7128,
                customer_lon=-74.0060,
                volume_m3=2.5,
                weight_kg=150.0
            ),
            OrderInput(
                order_id="ORD002",
                customer_lat=40.7589,
                customer_lon=-73.9851,
                volume_m3=1.8,
                weight_kg=120.0
            ),
            OrderInput(
                order_id="ORD003",
                customer_lat=40.6892,
                customer_lon=-74.0445,
                volume_m3=3.2,
                weight_kg=200.0
            )
        ]
    
    @pytest.fixture
    def sample_vehicles(self):
        """Create sample vehicles for testing."""
        return [
            VehicleInput(
                vehicle_id="VEH001",
                max_volume_m3=10.0,
                max_weight_kg=500.0,
                cost_per_km=2.5
            ),
            VehicleInput(
                vehicle_id="VEH002",
                max_volume_m3=8.0,
                max_weight_kg=400.0,
                cost_per_km=2.0
            )
        ]
    
    @pytest.mark.asyncio
    async def test_csv_processing(self, service):
        """Test CSV input processing."""
        csv_content = """
orders
order_id,customer_lat,customer_lon,volume_m3,weight_kg
ORD001,40.7128,-74.0060,2.5,150.0
ORD002,40.7589,-73.9851,1.8,120.0

vehicles
vehicle_id,max_volume_m3,max_weight_kg,cost_per_km
VEH001,10.0,500.0,2.5
VEH002,8.0,400.0,2.0
"""
        
        result = await service.process_orders_and_vehicles_csv(csv_content)
        
        assert isinstance(result, OrdersAndVehiclesCSV)
        assert len(result.orders) == 2
        assert len(result.vehicles) == 2
        assert result.orders[0].order_id == "ORD001"
        assert result.vehicles[0].vehicle_id == "VEH001"
    
    @pytest.mark.asyncio
    async def test_csv_processing_invalid_format(self, service):
        """Test CSV processing with invalid format."""
        invalid_csv = "invalid,csv,format"
        
        with pytest.raises(ValueError):
            await service.process_orders_and_vehicles_csv(invalid_csv)
    
    @pytest.mark.asyncio
    async def test_greedy_algorithm(self, service, sample_orders, sample_vehicles):
        """Test greedy optimization algorithm."""
        solution = await service.optimize_routes(
            orders=sample_orders,
            vehicles=sample_vehicles,
            algorithm="greedy",
            depot_lat=40.7831,
            depot_lon=-73.9712
        )
        
        assert solution.algorithm_used == "greedy"
        assert solution.total_distance >= 0
        assert solution.total_cost >= 0
        assert solution.num_orders_assigned <= len(sample_orders)
        assert solution.optimization_time_seconds > 0
        assert len(solution.vehicle_utilization) == len(sample_vehicles)
    
    @pytest.mark.asyncio
    async def test_nearest_neighbor_algorithm(self, service, sample_orders, sample_vehicles):
        """Test nearest neighbor optimization algorithm."""
        solution = await service.optimize_routes(
            orders=sample_orders,
            vehicles=sample_vehicles,
            algorithm="nearest_neighbor",
            depot_lat=40.7831,
            depot_lon=-73.9712
        )
        
        assert solution.algorithm_used == "nearest_neighbor"
        assert solution.total_distance >= 0
        assert solution.total_cost >= 0
        assert solution.num_orders_assigned <= len(sample_orders)
        assert solution.optimization_time_seconds > 0
    
    @pytest.mark.asyncio
    async def test_invalid_algorithm(self, service, sample_orders, sample_vehicles):
        """Test optimization with invalid algorithm."""
        with pytest.raises(ValueError):
            await service.optimize_routes(
                orders=sample_orders,
                vehicles=sample_vehicles,
                algorithm="invalid_algorithm",
                depot_lat=40.7831,
                depot_lon=-73.9712
            )
    
    @pytest.mark.asyncio
    async def test_capacity_validation(self, service):
        """Test vehicle capacity constraint validation."""
        # Create orders that exceed vehicle capacity
        orders = [
            OrderInput(order_id="ORD001", customer_lat=40.7128, customer_lon=-74.0060, volume_m3=6.0, weight_kg=300.0),
            OrderInput(order_id="ORD002", customer_lat=40.7589, customer_lon=-73.9851, volume_m3=5.0, weight_kg=250.0)
        ]
        
        vehicles = [
            VehicleInput(vehicle_id="VEH001", max_volume_m3=8.0, max_weight_kg=400.0, cost_per_km=2.5)
        ]
        
        # Create assignments that would overload the vehicle
        assignments = [
            RouteAssignment(vehicle_id="VEH001", order_id="ORD001", sequence=1, distance_km=5.0, total_cost=12.5),
            RouteAssignment(vehicle_id="VEH001", order_id="ORD002", sequence=2, distance_km=3.0, total_cost=7.5)
        ]
        
        utilizations = await service.validate_capacity_constraints(assignments, orders, vehicles)
        
        assert len(utilizations) == 1
        utilization = utilizations[0]
        assert utilization.vehicle_id == "VEH001"
        assert utilization.total_volume_used_m3 == 11.0  # 6.0 + 5.0
        assert utilization.total_weight_used_kg == 550.0  # 300.0 + 250.0
        assert utilization.is_overloaded == True  # Exceeds both volume and weight limits
    
    @pytest.mark.asyncio
    async def test_distance_matrix_calculation(self, service, sample_orders):
        """Test distance matrix calculation."""
        depot_lat, depot_lon = 40.7831, -73.9712
        
        distance_matrix = await service._calculate_distance_matrix(sample_orders, depot_lat, depot_lon)
        
        # Check matrix structure
        assert "depot" in distance_matrix
        for order in sample_orders:
            assert order.order_id in distance_matrix
            assert order.order_id in distance_matrix["depot"]
        
        # Check self-distances are zero
        assert distance_matrix["depot"]["depot"] == 0.0
        for order in sample_orders:
            assert distance_matrix[order.order_id][order.order_id] == 0.0
        
        # Check distances are positive
        for order in sample_orders:
            assert distance_matrix["depot"][order.order_id] > 0
    
    @pytest.mark.asyncio
    async def test_haversine_distance_calculation(self, service):
        """Test haversine distance calculation."""
        # Test known distance (approximately)
        # NYC to Central Park should be around 5-6 km
        distance = service._haversine_distance(40.7128, -74.0060, 40.7589, -73.9851)
        
        assert isinstance(distance, float)
        assert 4.0 < distance < 8.0  # Reasonable range for this distance
    
    @pytest.mark.asyncio
    async def test_route_assignments_csv_generation(self, service):
        """Test CSV output generation."""
        assignments = [
            RouteAssignment(vehicle_id="VEH001", order_id="ORD001", sequence=1, distance_km=5.2, total_cost=13.0),
            RouteAssignment(vehicle_id="VEH001", order_id="ORD002", sequence=2, distance_km=3.8, total_cost=9.5),
            RouteAssignment(vehicle_id="VEH002", order_id="ORD003", sequence=1, distance_km=4.1, total_cost=8.2)
        ]
        
        csv_output = await service.generate_route_assignments_csv(assignments)
        
        assert isinstance(csv_output, str)
        assert "vehicle_id,order_id,sequence,distance_km,total_cost" in csv_output
        assert "VEH001,ORD001,1,5.2,13.0" in csv_output
        assert "VEH002,ORD003,1,4.1,8.2" in csv_output
    
    @pytest.mark.asyncio
    async def test_optimization_statistics(self, service, sample_orders, sample_vehicles):
        """Test optimization statistics generation."""
        solution = await service.optimize_routes(
            orders=sample_orders,
            vehicles=sample_vehicles,
            algorithm="greedy",
            depot_lat=40.7831,
            depot_lon=-73.9712
        )
        
        stats = await service.get_optimization_statistics(solution)
        
        assert "solution_id" in stats
        assert "algorithm_used" in stats
        assert "total_distance_km" in stats
        assert "total_cost" in stats
        assert "assignment_rate_percent" in stats
        assert "average_vehicle_utilization" in stats
        assert stats["algorithm_used"] == "greedy"
        assert 0 <= stats["assignment_rate_percent"] <= 100


class TestDistributionAPI:
    """Test distribution API endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/distribution/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "distribution"
        assert data["status"] == "healthy"
    
    def test_optimize_routes_endpoint(self):
        """Test route optimization endpoint."""
        request_data = {
            "request_id": "REQ001",
            "orders": [
                {
                    "order_id": "ORD001",
                    "customer_lat": 40.7128,
                    "customer_lon": -74.0060,
                    "volume_m3": 2.5,
                    "weight_kg": 150.0
                }
            ],
            "vehicles": [
                {
                    "vehicle_id": "VEH001",
                    "max_volume_m3": 10.0,
                    "max_weight_kg": 500.0,
                    "cost_per_km": 2.5
                }
            ],
            "algorithm_preference": "greedy"
        }
        
        response = client.post("/distribution/optimize-routes", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["request_id"] == "REQ001"
        assert data["optimization_status"] == "success"
        assert "solution" in data
        assert "routes" in data
        assert "assignments" in data
    
    def test_optimize_routes_invalid_request(self):
        """Test route optimization with invalid request."""
        invalid_request = {
            "request_id": "REQ002",
            "orders": [],  # Empty orders list
            "vehicles": [
                {
                    "vehicle_id": "VEH001",
                    "max_volume_m3": 10.0,
                    "max_weight_kg": 500.0,
                    "cost_per_km": 2.5
                }
            ]
        }
        
        response = client.post("/distribution/optimize-routes", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_optimize_csv_endpoint(self):
        """Test CSV optimization endpoint."""
        csv_content = """orders
order_id,customer_lat,customer_lon,volume_m3,weight_kg
ORD001,40.7128,-74.0060,2.5,150.0

vehicles
vehicle_id,max_volume_m3,max_weight_kg,cost_per_km
VEH001,10.0,500.0,2.5"""
        
        files = {"file": ("test.csv", io.StringIO(csv_content), "text/csv")}
        response = client.post("/distribution/optimize-csv", files=files)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "vehicle_id,order_id,sequence,distance_km,total_cost" in response.text
    
    def test_validate_capacity_endpoint(self):
        """Test capacity validation endpoint."""
        request_data = {
            "assignments": [
                {
                    "vehicle_id": "VEH001",
                    "order_id": "ORD001",
                    "sequence": 1,
                    "distance_km": 5.0,
                    "total_cost": 12.5
                }
            ],
            "orders": [
                {
                    "order_id": "ORD001",
                    "customer_lat": 40.7128,
                    "customer_lon": -74.0060,
                    "volume_m3": 2.5,
                    "weight_kg": 150.0
                }
            ],
            "vehicles": [
                {
                    "vehicle_id": "VEH001",
                    "max_volume_m3": 10.0,
                    "max_weight_kg": 500.0,
                    "cost_per_km": 2.5
                }
            ]
        }
        
        response = client.post("/distribution/validate-capacity", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["vehicle_id"] == "VEH001"
        assert data[0]["is_overloaded"] == False
    
    def test_create_distribution_center(self):
        """Test distribution center creation endpoint."""
        center_data = {
            "center_id": "DC001",
            "name": "Test Center",
            "latitude": 40.7831,
            "longitude": -73.9712,
            "max_vehicles": 20,
            "operating_hours_start": "06:00",
            "operating_hours_end": "22:00",
            "is_active": True
        }
        
        response = client.post("/distribution/centers", json=center_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["center_id"] == "DC001"
        assert data["name"] == "Test Center"
    
    def test_list_distribution_centers(self):
        """Test distribution centers listing endpoint."""
        response = client.get("/distribution/centers")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_performance_summary(self):
        """Test performance summary endpoint."""
        response = client.get("/distribution/performance/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_routes" in data
        assert "total_distance_km" in data
        assert "total_cost" in data
        assert "average_vehicle_utilization" in data
    
    def test_vehicle_performance(self):
        """Test vehicle performance endpoint."""
        response = client.get("/distribution/performance/vehicles")
        assert response.status_code == 200
        
        data = response.json()
        assert "vehicles" in data
        assert "analysis_period" in data


class TestDistributionIntegration:
    """Integration tests for distribution functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow from CSV to results."""
        service = DistributionService()
        
        # Sample CSV input
        csv_content = """
orders
order_id,customer_lat,customer_lon,volume_m3,weight_kg
ORD001,40.7128,-74.0060,2.5,150.0
ORD002,40.7589,-73.9851,1.8,120.0
ORD003,40.6892,-74.0445,3.2,200.0

vehicles
vehicle_id,max_volume_m3,max_weight_kg,cost_per_km
VEH001,10.0,500.0,2.5
VEH002,8.0,400.0,2.0
"""
        
        # Process complete workflow
        solution, output_csv = await service.process_distribution_optimization(
            csv_content=csv_content,
            algorithm="greedy",
            depot_lat=40.7831,
            depot_lon=-73.9712
        )
        
        # Verify solution
        assert solution.num_orders_assigned > 0
        assert solution.total_distance > 0
        assert solution.total_cost > 0
        
        # Verify CSV output
        assert isinstance(output_csv, str)
        assert "vehicle_id,order_id,sequence,distance_km,total_cost" in output_csv
        
        # Count lines in output (header + data rows)
        lines = output_csv.strip().split('\n')
        assert len(lines) >= 2  # At least header + one data row
    
    @pytest.mark.asyncio
    async def test_algorithm_comparison(self):
        """Test comparison between different optimization algorithms."""
        service = DistributionService()
        
        # Create test data
        orders = [
            OrderInput(order_id=f"ORD{i:03d}", customer_lat=40.7 + (i * 0.01), 
                      customer_lon=-74.0 + (i * 0.01), volume_m3=2.0, weight_kg=100.0)
            for i in range(1, 6)
        ]
        
        vehicles = [
            VehicleInput(vehicle_id="VEH001", max_volume_m3=15.0, max_weight_kg=800.0, cost_per_km=2.5),
            VehicleInput(vehicle_id="VEH002", max_volume_m3=12.0, max_weight_kg=600.0, cost_per_km=2.0)
        ]
        
        # Test both algorithms
        greedy_solution = await service.optimize_routes(orders, vehicles, "greedy", 40.7831, -73.9712)
        nn_solution = await service.optimize_routes(orders, vehicles, "nearest_neighbor", 40.7831, -73.9712)
        
        # Both should produce valid solutions
        assert greedy_solution.num_orders_assigned > 0
        assert nn_solution.num_orders_assigned > 0
        assert greedy_solution.algorithm_used == "greedy"
        assert nn_solution.algorithm_used == "nearest_neighbor"
        
        # Solutions may differ but should be reasonable
        assert greedy_solution.total_distance > 0
        assert nn_solution.total_distance > 0