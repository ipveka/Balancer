"""
Pytest configuration and fixtures for the Balancer platform test suite.

This module provides comprehensive test configuration including FastAPI test client setup,
mock data management, database configuration, and shared test utilities for all domain modules.
"""

import asyncio
import json
import os
import tempfile
from typing import Dict, Any, List, Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
import pandas as pd

# Import main application
from main import app
from config import settings, Settings

# Import domain services for mocking
from supply.service import SupplyService
from inventory.service import InventoryService
from demand.service import DemandService
from distribution.service import DistributionService

# Import models for test data creation
from supply.models import (
    ProcurementDataInput, ManufacturingDataInput,
    ProcurementRecommendation, ManufacturingRecommendation
)
from inventory.models import (
    InventoryStatusInput, InventoryRecommendation,
    InventoryItem, InventoryTransaction
)
from demand.models import (
    DemandDataInput, ForecastOutput,
    DemandRecord, DemandAnalytics
)
from distribution.models import (
    OrderInput, VehicleInput, RouteAssignment,
    DistributionCenter, DeliveryRoute
)

# Import utilities
from utils.dummy_data import (
    generate_procurement_data, generate_manufacturing_data,
    generate_inventory_data, generate_demand_data, generate_distribution_data
)


# Test configuration
@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Test-specific settings configuration.
    
    Returns:
        Settings instance configured for testing
    """
    # Override settings for testing
    test_config = Settings(
        environment="testing",
        api=Settings.APIConfig(
            debug=True,
            title="Balancer API - Test",
            cors_origins=["*"]
        ),
        ai=Settings.AIConfig(
            forecast_frequency="weekly",
            forecast_horizon=4,  # Shorter for faster tests
            default_service_level=0.95,
            vrp_algorithm="greedy",
            optimization_iterations=10  # Fewer iterations for faster tests
        )
    )
    return test_config


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    
    This fixture ensures that async tests run properly in pytest.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# FastAPI test client fixtures
@pytest.fixture
def client(test_settings: Settings) -> Generator[TestClient, None, None]:
    """
    Synchronous FastAPI test client.
    
    Args:
        test_settings: Test configuration settings
        
    Yields:
        TestClient instance for synchronous API testing
    """
    with patch('config.settings', test_settings):
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
async def async_client(test_settings: Settings) -> AsyncGenerator[AsyncClient, None]:
    """
    Asynchronous FastAPI test client.
    
    Args:
        test_settings: Test configuration settings
        
    Yields:
        AsyncClient instance for asynchronous API testing
    """
    with patch('config.settings', test_settings):
        async with AsyncClient(app=app, base_url="http://test") as async_test_client:
            yield async_test_client


# Mock service fixtures
@pytest.fixture
def mock_supply_service() -> Mock:
    """
    Mock supply service for testing.
    
    Returns:
        Mock SupplyService instance with common methods mocked
    """
    mock_service = Mock(spec=SupplyService)
    mock_service.process_procurement_csv = AsyncMock()
    mock_service.process_manufacturing_csv = AsyncMock()
    mock_service.export_procurement_recommendations_csv = AsyncMock()
    mock_service.export_manufacturing_recommendations_csv = AsyncMock()
    return mock_service


@pytest.fixture
def mock_inventory_service() -> Mock:
    """
    Mock inventory service for testing.
    
    Returns:
        Mock InventoryService instance with common methods mocked
    """
    mock_service = Mock(spec=InventoryService)
    mock_service.process_inventory_csv = AsyncMock()
    mock_service.export_inventory_recommendations_csv = AsyncMock()
    mock_service.create_inventory_item = AsyncMock()
    mock_service.create_inventory_transaction = AsyncMock()
    mock_service.get_optimization_summary = AsyncMock()
    return mock_service


@pytest.fixture
def mock_demand_service() -> Mock:
    """
    Mock demand service for testing.
    
    Returns:
        Mock DemandService instance with common methods mocked
    """
    mock_service = Mock(spec=DemandService)
    mock_service.process_demand_csv = AsyncMock()
    mock_service.export_forecast_csv = AsyncMock()
    mock_service.get_model_summary = AsyncMock()
    mock_service.trained_models = {}
    mock_service.default_forecast_horizon_weeks = 12
    mock_service.confidence_level = 0.95
    return mock_service


@pytest.fixture
def mock_distribution_service() -> Mock:
    """
    Mock distribution service for testing.
    
    Returns:
        Mock DistributionService instance with common methods mocked
    """
    mock_service = Mock(spec=DistributionService)
    mock_service.optimize_routes = AsyncMock()
    mock_service.process_distribution_optimization = AsyncMock()
    mock_service.validate_capacity_constraints = AsyncMock()
    mock_service._reconstruct_assignments_from_solution = AsyncMock()
    return mock_service


# Test data fixtures
@pytest.fixture
def sample_procurement_data() -> List[ProcurementDataInput]:
    """
    Sample procurement data for testing.
    
    Returns:
        List of ProcurementDataInput objects
    """
    return [
        ProcurementDataInput(
            sku="WIDGET-001",
            current_inventory=100,
            forecast_demand_4weeks=500,
            safety_stock=50,
            min_order_qty=100,
            supplier_id="SUP-001",
            unit_cost=10.50
        ),
        ProcurementDataInput(
            sku="GADGET-002",
            current_inventory=25,
            forecast_demand_4weeks=200,
            safety_stock=30,
            min_order_qty=50,
            supplier_id="SUP-002",
            unit_cost=25.75
        )
    ]


@pytest.fixture
def sample_manufacturing_data() -> List[ManufacturingDataInput]:
    """
    Sample manufacturing data for testing.
    
    Returns:
        List of ManufacturingDataInput objects
    """
    return [
        ManufacturingDataInput(
            sku="WIDGET-001",
            current_inventory=100,
            forecast_demand_4weeks=500,
            safety_stock=50,
            batch_size=200,
            production_time_days=5,
            unit_cost=8.50
        ),
        ManufacturingDataInput(
            sku="GADGET-002",
            current_inventory=25,
            forecast_demand_4weeks=200,
            safety_stock=30,
            batch_size=100,
            production_time_days=3,
            unit_cost=18.75
        )
    ]


@pytest.fixture
def sample_inventory_data() -> List[InventoryStatusInput]:
    """
    Sample inventory data for testing.
    
    Returns:
        List of InventoryStatusInput objects
    """
    return [
        InventoryStatusInput(
            sku="WIDGET-001",
            current_stock=150,
            lead_time_days=7,
            service_level_target=0.95,
            avg_weekly_demand=50.0,
            demand_std_dev=12.5
        ),
        InventoryStatusInput(
            sku="GADGET-002",
            current_stock=75,
            lead_time_days=14,
            service_level_target=0.90,
            avg_weekly_demand=25.0,
            demand_std_dev=8.0
        )
    ]


@pytest.fixture
def sample_demand_data() -> List[DemandDataInput]:
    """
    Sample demand data for testing.
    
    Returns:
        List of DemandDataInput objects
    """
    return [
        DemandDataInput(date="2024-01-01", sku="WIDGET-001", quantity=150),
        DemandDataInput(date="2024-01-02", sku="WIDGET-001", quantity=142),
        DemandDataInput(date="2024-01-03", sku="WIDGET-001", quantity=158),
        DemandDataInput(date="2024-01-01", sku="GADGET-002", quantity=85),
        DemandDataInput(date="2024-01-02", sku="GADGET-002", quantity=92),
        DemandDataInput(date="2024-01-03", sku="GADGET-002", quantity=78)
    ]


@pytest.fixture
def sample_orders_data() -> List[OrderInput]:
    """
    Sample orders data for testing.
    
    Returns:
        List of OrderInput objects
    """
    return [
        OrderInput(
            order_id="ORD-001",
            customer_lat=40.7128,
            customer_lon=-74.0060,
            volume_m3=2.5,
            weight_kg=150.0
        ),
        OrderInput(
            order_id="ORD-002",
            customer_lat=40.7589,
            customer_lon=-73.9851,
            volume_m3=1.8,
            weight_kg=120.0
        ),
        OrderInput(
            order_id="ORD-003",
            customer_lat=40.6892,
            customer_lon=-74.0445,
            volume_m3=3.2,
            weight_kg=200.0
        )
    ]


@pytest.fixture
def sample_vehicles_data() -> List[VehicleInput]:
    """
    Sample vehicles data for testing.
    
    Returns:
        List of VehicleInput objects
    """
    return [
        VehicleInput(
            vehicle_id="VEH-001",
            max_volume_m3=10.0,
            max_weight_kg=1000.0,
            cost_per_km=2.50
        ),
        VehicleInput(
            vehicle_id="VEH-002",
            max_volume_m3=15.0,
            max_weight_kg=1500.0,
            cost_per_km=3.00
        )
    ]


@pytest.fixture
def sample_distribution_center() -> DistributionCenter:
    """
    Sample distribution center for testing.
    
    Returns:
        DistributionCenter object
    """
    return DistributionCenter(
        center_id="DC-001",
        name="Main Distribution Center",
        latitude=40.7128,
        longitude=-74.0060,
        max_vehicles=10,
        operating_hours_start="08:00",
        operating_hours_end="18:00",
        is_active=True
    )


# CSV test data fixtures
@pytest.fixture
def sample_procurement_csv() -> str:
    """
    Sample procurement CSV data for testing.
    
    Returns:
        CSV string with procurement data
    """
    return """sku,current_inventory,forecast_demand_4weeks,safety_stock,min_order_qty,supplier_id,unit_cost
WIDGET-001,100,500,50,100,SUP-001,10.50
GADGET-002,25,200,30,50,SUP-002,25.75
TOOL-003,0,150,20,25,SUP-001,15.25"""


@pytest.fixture
def sample_manufacturing_csv() -> str:
    """
    Sample manufacturing CSV data for testing.
    
    Returns:
        CSV string with manufacturing data
    """
    return """sku,current_inventory,forecast_demand_4weeks,safety_stock,batch_size,production_time_days,unit_cost
WIDGET-001,100,500,50,200,5,8.50
GADGET-002,25,200,30,100,3,18.75
TOOL-003,0,150,20,50,7,12.25"""


@pytest.fixture
def sample_inventory_csv() -> str:
    """
    Sample inventory CSV data for testing.
    
    Returns:
        CSV string with inventory data
    """
    return """sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev
WIDGET-001,150,7,0.95,50.0,12.5
GADGET-002,75,14,0.90,25.0,8.0
TOOL-003,200,10,0.95,40.0,15.0
PART-004,50,21,0.85,15.0,5.0"""


@pytest.fixture
def sample_demand_csv() -> str:
    """
    Sample demand CSV data for testing.
    
    Returns:
        CSV string with demand data
    """
    return """date,sku,quantity
2024-01-01,WIDGET-001,150
2024-01-02,WIDGET-001,142
2024-01-03,WIDGET-001,158
2024-01-04,WIDGET-001,135
2024-01-05,WIDGET-001,167
2024-01-01,GADGET-002,85
2024-01-02,GADGET-002,92
2024-01-03,GADGET-002,78
2024-01-04,GADGET-002,88
2024-01-05,GADGET-002,95"""


@pytest.fixture
def sample_distribution_csv() -> str:
    """
    Sample distribution CSV data for testing.
    
    Returns:
        CSV string with orders and vehicles data
    """
    return """order_id,customer_lat,customer_lon,volume_m3,weight_kg,vehicle_id,max_volume_m3,max_weight_kg,cost_per_km
ORD-001,40.7128,-74.0060,2.5,150.0,VEH-001,10.0,1000.0,2.50
ORD-002,40.7589,-73.9851,1.8,120.0,VEH-001,10.0,1000.0,2.50
ORD-003,40.6892,-74.0445,3.2,200.0,VEH-002,15.0,1500.0,3.00"""


# File handling fixtures
@pytest.fixture
def temp_csv_file() -> Generator[str, None, None]:
    """
    Temporary CSV file for testing file uploads.
    
    Yields:
        Path to temporary CSV file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("test,data\n1,2\n3,4\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """
    Temporary directory for testing file operations.
    
    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# Database mock fixtures (for future database integration)
@pytest.fixture
def mock_database():
    """
    Mock database connection for testing.
    
    Returns:
        Mock database connection object
    """
    mock_db = Mock()
    mock_db.execute = AsyncMock()
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchrow = AsyncMock(return_value=None)
    mock_db.fetchval = AsyncMock(return_value=None)
    return mock_db


# Test utilities
@pytest.fixture
def assert_csv_structure():
    """
    Utility function for asserting CSV structure in tests.
    
    Returns:
        Function that validates CSV structure
    """
    def _assert_csv_structure(csv_content: str, expected_columns: List[str], min_rows: int = 1):
        """
        Assert that CSV content has expected structure.
        
        Args:
            csv_content: CSV content as string
            expected_columns: List of expected column names
            min_rows: Minimum number of data rows expected
        """
        lines = csv_content.strip().split('\n')
        assert len(lines) >= min_rows + 1, f"Expected at least {min_rows + 1} lines, got {len(lines)}"
        
        # Check header
        header = lines[0].split(',')
        assert header == expected_columns, f"Expected columns {expected_columns}, got {header}"
        
        # Check data rows
        for i, line in enumerate(lines[1:], 1):
            values = line.split(',')
            assert len(values) == len(expected_columns), f"Row {i} has {len(values)} values, expected {len(expected_columns)}"
    
    return _assert_csv_structure


@pytest.fixture
def assert_api_response():
    """
    Utility function for asserting API response structure.
    
    Returns:
        Function that validates API response structure
    """
    def _assert_api_response(response_data: Dict[str, Any], required_fields: List[str]):
        """
        Assert that API response has required fields.
        
        Args:
            response_data: Response data dictionary
            required_fields: List of required field names
        """
        for field in required_fields:
            assert field in response_data, f"Required field '{field}' missing from response"
            assert response_data[field] is not None, f"Required field '{field}' is None"
    
    return _assert_api_response


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """
    Utility for measuring test performance.
    
    Returns:
        Context manager for timing operations
    """
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def _timer():
        start_time = time.time()
        yield
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.3f} seconds")
    
    return _timer


# Parametrized test data
@pytest.fixture(params=["greedy", "nearest_neighbor"])
def vrp_algorithm(request):
    """
    Parametrized fixture for VRP algorithms.
    
    Returns:
        VRP algorithm name for parametrized tests
    """
    return request.param


@pytest.fixture(params=[0.90, 0.95, 0.99])
def service_level(request):
    """
    Parametrized fixture for service levels.
    
    Returns:
        Service level value for parametrized tests
    """
    return request.param


@pytest.fixture(params=[4, 8, 12])
def forecast_horizon(request):
    """
    Parametrized fixture for forecast horizons.
    
    Returns:
        Forecast horizon in weeks for parametrized tests
    """
    return request.param


# Integration test fixtures
@pytest.fixture
def integration_test_data():
    """
    Comprehensive test data for integration tests.
    
    Returns:
        Dictionary with test data for all modules
    """
    return {
        "supply": {
            "procurement": generate_procurement_data(num_items=10),
            "manufacturing": generate_manufacturing_data(num_items=10)
        },
        "inventory": generate_inventory_data(num_items=15),
        "demand": generate_demand_data(num_skus=5, num_days=30),
        "distribution": generate_distribution_data(num_orders=20, num_vehicles=5)
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """
    Automatic cleanup fixture that runs after each test.
    
    Ensures test environment is clean between tests.
    """
    yield
    
    # Cleanup any test artifacts
    # (Clear caches, reset mocks, etc.)
    pass


# Logging configuration for tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """
    Configure logging for tests.
    
    Reduces log noise during test execution while maintaining
    error visibility.
    """
    import logging
    
    # Set logging level to WARNING to reduce test noise
    logging.getLogger().setLevel(logging.WARNING)
    
    # Keep error logs visible
    logging.getLogger("main").setLevel(logging.ERROR)
    logging.getLogger("supply").setLevel(logging.ERROR)
    logging.getLogger("inventory").setLevel(logging.ERROR)
    logging.getLogger("demand").setLevel(logging.ERROR)
    logging.getLogger("distribution").setLevel(logging.ERROR)
    
    yield
    
    # Reset logging after tests
    logging.getLogger().setLevel(logging.INFO)


# Pytest configuration
def pytest_configure(config):
    """
    Pytest configuration hook.
    
    Configures pytest with custom markers and settings.
    """
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers based on test names.
    
    Automatically adds markers to tests based on naming conventions.
    """
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in item.nodeid or "perf" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.nodeid for keyword in ["forecast", "optimization", "ml"]):
            item.add_marker(pytest.mark.slow)
        
        # Add unit marker to unit tests (default)
        if not any(marker.name in ["integration", "performance"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)