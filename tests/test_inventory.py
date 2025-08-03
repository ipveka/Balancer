"""
Inventory Module Tests

Comprehensive test suite for the inventory management module including
API endpoint tests, service logic tests, safety stock calculations,
and error handling validation.
"""

import pytest
import asyncio
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json
import io

from inventory.api import router as inventory_router
from inventory.service import InventoryService, process_inventory_csv_file
from inventory.models import (
    InventoryOptimizationRequest, InventoryItemRequest, InventoryTransactionRequest,
    InventoryStatusInput, TransactionType, InventoryStatus, RecommendedAction,
    InventoryException, InsufficientInventoryException, StockCalculationException
)
from utils.helpers import CSVProcessingError, DataValidationError


# Test fixtures
@pytest.fixture
def app():
    """Create FastAPI app with inventory router for testing."""
    app = FastAPI()
    app.include_router(inventory_router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def inventory_service():
    """Create inventory service instance."""
    return InventoryService()


@pytest.fixture
def sample_inventory_csv():
    """Sample inventory CSV data for testing."""
    return """sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev
WIDGET-001,150,7,0.95,50.0,12.5
GADGET-002,25,14,0.90,35.0,8.0
TOOL-003,300,10,0.95,40.0,15.0
COMPONENT-004,75,21,0.85,25.0,6.0
PART-005,10,7,0.99,20.0,4.0"""


@pytest.fixture
def low_stock_csv():
    """CSV data with low stock scenarios."""
    return """sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev
LOW-STOCK-001,5,7,0.95,20.0,5.0
CRITICAL-002,2,14,0.90,15.0,3.0
URGENT-003,0,10,0.95,25.0,8.0"""


@pytest.fixture
def invalid_csv():
    """Invalid CSV data for error testing."""
    return """invalid,headers,format
some,data,here"""


@pytest.fixture
def missing_columns_csv():
    """CSV with missing required columns."""
    return """sku,current_stock
WIDGET-001,100"""


@pytest.fixture
def sample_inventory_item():
    """Sample inventory item data."""
    return {
        "sku": "TEST-ITEM-001",
        "description": "Test inventory item",
        "current_stock": 100,
        "location": "WAREHOUSE-A",
        "unit_cost": 15.50,
        "reorder_point": 50,
        "safety_stock": 25,
        "max_stock": 300
    }


@pytest.fixture
def sample_transaction():
    """Sample inventory transaction data."""
    return {
        "sku": "TEST-ITEM-001",
        "transaction_type": TransactionType.INBOUND,
        "quantity": 50,
        "unit_cost": 15.50,
        "reference_number": "PO-TEST-001",
        "location": "WAREHOUSE-A",
        "notes": "Test transaction",
        "created_by": "test_user"
    }


# API Endpoint Tests

class TestInventoryAPI:
    """Test class for inventory API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/inventory/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["module"] == "inventory"
        assert "supported_operations" in data
        assert "optimization" in data["supported_operations"]
        assert "item_management" in data["supported_operations"]
        assert "transaction_recording" in data["supported_operations"]
    
    def test_get_recommendation_actions(self, client):
        """Test get recommendation actions endpoint."""
        response = client.get("/inventory/actions")
        assert response.status_code == 200
        
        data = response.json()
        assert "supported_actions" in data
        assert len(data["supported_actions"]) == 5
        
        actions = [action["action"] for action in data["supported_actions"]]
        assert "REORDER" in actions
        assert "URGENT_REORDER" in actions
        assert "SUFFICIENT_STOCK" in actions
        assert "EXCESS_STOCK" in actions
        assert "REVIEW_REQUIRED" in actions
    
    def test_get_inventory_status(self, client):
        """Test get inventory status endpoint."""
        response = client.get("/inventory/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["module"] == "inventory"
        assert data["status"] == "operational"
        assert "features" in data
        assert "configuration" in data
        assert data["features"]["inventory_optimization"] is True
        assert data["features"]["safety_stock_calculation"] is True
    
    def test_download_inventory_template(self, client):
        """Test inventory template download."""
        response = client.get("/inventory/template")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "inventory_template.csv" in response.headers["content-disposition"]
        
        # Verify CSV content
        content = response.content.decode('utf-8')
        assert "sku,current_stock,lead_time_days,service_level_target" in content
        assert "WIDGET-001" in content
    
    def test_optimize_inventory(self, client, sample_inventory_csv):
        """Test inventory optimization endpoint."""
        request_data = {
            "csv_data": sample_inventory_csv,
            "optimization_params": {"test_param": "value"}
        }
        
        response = client.post("/inventory/optimize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["recommendations_count"] == 5
        assert "csv_output" in data
        assert "processing_time_seconds" in data
        assert "optimization_summary" in data
    
    def test_optimize_inventory_invalid_csv(self, client, invalid_csv):
        """Test optimization with invalid CSV data."""
        request_data = {
            "csv_data": invalid_csv,
            "optimization_params": {}
        }
        
        response = client.post("/inventory/optimize", json=request_data)
        assert response.status_code == 400
        assert "CSV processing failed" in response.json()["detail"]
    
    def test_optimize_inventory_missing_columns(self, client, missing_columns_csv):
        """Test optimization with missing required columns."""
        request_data = {
            "csv_data": missing_columns_csv,
            "optimization_params": {}
        }
        
        response = client.post("/inventory/optimize", json=request_data)
        assert response.status_code == 400
        assert "Missing required columns" in response.json()["detail"]
    
    def test_upload_inventory_csv(self, client, sample_inventory_csv):
        """Test inventory CSV file upload."""
        files = {"file": ("test.csv", io.StringIO(sample_inventory_csv), "text/csv")}
        
        response = client.post("/inventory/upload", files=files)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "inventory_recommendations.csv" in response.headers["content-disposition"]
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        files = {"file": ("test.txt", io.StringIO("test content"), "text/plain")}
        
        response = client.post("/inventory/upload", files=files)
        assert response.status_code == 400
        assert "File must be a CSV file" in response.json()["detail"]
    
    def test_create_inventory_item(self, client, sample_inventory_item):
        """Test inventory item creation endpoint."""
        response = client.post("/inventory/items", json=sample_inventory_item)
        assert response.status_code == 200
        
        data = response.json()
        assert data["sku"] == sample_inventory_item["sku"]
        assert data["description"] == sample_inventory_item["description"]
        assert data["current_stock"] == sample_inventory_item["current_stock"]
        assert data["available_stock"] == sample_inventory_item["current_stock"]  # No reserved stock
        assert data["location"] == sample_inventory_item["location"]
    
    def test_get_inventory_items(self, client):
        """Test get inventory items endpoint."""
        response = client.get("/inventory/items")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # Mock implementation returns empty list
        assert len(data) == 0
    
    def test_get_inventory_item_not_found(self, client):
        """Test get specific inventory item (not found)."""
        response = client.get("/inventory/items/NONEXISTENT-001")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_create_inventory_transaction(self, client, sample_transaction):
        """Test inventory transaction creation endpoint."""
        response = client.post("/inventory/transactions", json=sample_transaction)
        assert response.status_code == 200
        
        data = response.json()
        assert data["sku"] == sample_transaction["sku"]
        assert data["transaction_type"] == sample_transaction["transaction_type"]
        assert data["quantity"] == sample_transaction["quantity"]
        assert data["location"] == sample_transaction["location"]
        assert "transaction_id" in data
    
    def test_get_inventory_transactions(self, client):
        """Test get inventory transactions endpoint."""
        response = client.get("/inventory/transactions")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # Mock implementation returns empty list
        assert len(data) == 0
    
    def test_get_inventory_analytics(self, client):
        """Test get inventory analytics endpoint."""
        response = client.get("/inventory/analytics/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_items" in data
        assert "total_value" in data
        assert "items_below_reorder_point" in data
        assert "stockout_risk_items" in data


# Service Layer Tests

class TestInventoryService:
    """Test class for inventory service layer."""
    
    @pytest.mark.asyncio
    async def test_process_inventory_csv_success(self, inventory_service, sample_inventory_csv):
        """Test successful inventory CSV processing."""
        result = await inventory_service.process_inventory_csv(sample_inventory_csv)
        
        assert result.total_recommendations == 5
        assert len(result.recommendations) == 5
        
        # Check first recommendation
        rec = result.recommendations[0]
        assert rec.sku == "WIDGET-001"
        assert rec.safety_stock >= 0
        assert rec.reorder_point >= rec.safety_stock
        assert rec.confidence_score >= 0 and rec.confidence_score <= 1
        assert rec.recommended_action in [action.value for action in RecommendedAction]
    
    @pytest.mark.asyncio
    async def test_process_inventory_csv_low_stock(self, inventory_service, low_stock_csv):
        """Test inventory CSV processing with low stock scenarios."""
        result = await inventory_service.process_inventory_csv(low_stock_csv)
        
        assert result.total_recommendations == 3
        
        # Check that low stock items get urgent recommendations
        urgent_items = [rec for rec in result.recommendations 
                       if rec.recommended_action == RecommendedAction.URGENT_REORDER]
        assert len(urgent_items) >= 1  # At least one urgent item
    
    @pytest.mark.asyncio
    async def test_process_inventory_csv_invalid_data(self, inventory_service, invalid_csv):
        """Test inventory CSV processing with invalid data."""
        with pytest.raises(CSVProcessingError):
            await inventory_service.process_inventory_csv(invalid_csv)
    
    @pytest.mark.asyncio
    async def test_calculate_safety_stock(self, inventory_service):
        """Test safety stock calculation logic."""
        item = InventoryStatusInput(
            sku="TEST-001",
            current_stock=100,
            lead_time_days=14,
            service_level_target=0.95,
            avg_weekly_demand=50.0,
            demand_std_dev=12.5
        )
        
        result = await inventory_service._calculate_safety_stock(item)
        
        assert result.sku == "TEST-001"
        assert result.service_level == 0.95
        assert result.lead_time_days == 14
        assert result.z_score > 0  # Should be positive for 95% service level
        assert result.calculated_safety_stock >= 0
        assert result.reorder_point >= result.calculated_safety_stock
        
        # Verify calculation logic
        lead_time_weeks = 14 / 7.0
        expected_avg_demand = 50.0 * lead_time_weeks
        assert abs(result.avg_demand_during_lead_time - expected_avg_demand) < 0.1
    
    @pytest.mark.asyncio
    async def test_calculate_safety_stock_high_service_level(self, inventory_service):
        """Test safety stock calculation with high service level."""
        item = InventoryStatusInput(
            sku="HIGH-SL-001",
            current_stock=100,
            lead_time_days=14,
            service_level_target=0.99,  # 99% service level
            avg_weekly_demand=50.0,
            demand_std_dev=12.5
        )
        
        result = await inventory_service._calculate_safety_stock(item)
        
        # Higher service level should result in higher Z-score and safety stock
        assert result.z_score > 2.0  # 99% service level Z-score
        assert result.calculated_safety_stock > 20  # Should be substantial
    
    @pytest.mark.asyncio
    async def test_calculate_safety_stock_low_variability(self, inventory_service):
        """Test safety stock calculation with low demand variability."""
        item = InventoryStatusInput(
            sku="LOW-VAR-001",
            current_stock=100,
            lead_time_days=14,
            service_level_target=0.95,
            avg_weekly_demand=50.0,
            demand_std_dev=2.5  # Low variability
        )
        
        result = await inventory_service._calculate_safety_stock(item)
        
        # Low variability should result in lower safety stock
        assert result.demand_variability < 0.1  # CV should be low
        assert result.calculated_safety_stock < 10  # Should be relatively low
    
    @pytest.mark.asyncio
    async def test_create_inventory_item(self, inventory_service, sample_inventory_item):
        """Test inventory item creation."""
        item = await inventory_service.create_inventory_item(sample_inventory_item)
        
        assert item.sku == sample_inventory_item["sku"]
        assert item.description == sample_inventory_item["description"]
        assert item.current_stock == sample_inventory_item["current_stock"]
        assert item.available_stock == sample_inventory_item["current_stock"]  # No reserved stock
        assert item.location == sample_inventory_item["location"]
        assert item.status == InventoryStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_create_inventory_item_with_reserved_stock(self, inventory_service):
        """Test inventory item creation with reserved stock."""
        item_data = {
            "sku": "RESERVED-001",
            "description": "Item with reserved stock",
            "current_stock": 100,
            "reserved_stock": 25,
            "location": "WAREHOUSE-A",
            "unit_cost": 10.00,
            "reorder_point": 50,
            "safety_stock": 30
        }
        
        item = await inventory_service.create_inventory_item(item_data)
        
        assert item.current_stock == 100
        assert item.reserved_stock == 25
        assert item.available_stock == 75  # 100 - 25
    
    @pytest.mark.asyncio
    async def test_create_inventory_transaction(self, inventory_service, sample_transaction):
        """Test inventory transaction creation."""
        transaction = await inventory_service.create_inventory_transaction(sample_transaction)
        
        assert transaction.sku == sample_transaction["sku"]
        assert transaction.transaction_type == sample_transaction["transaction_type"]
        assert transaction.quantity == sample_transaction["quantity"]
        assert transaction.location == sample_transaction["location"]
        assert transaction.created_by == sample_transaction["created_by"]
        assert transaction.transaction_id is not None
    
    @pytest.mark.asyncio
    async def test_update_stock_levels_inbound(self, inventory_service, sample_inventory_item):
        """Test stock level updates for inbound transactions."""
        # Create inventory item
        item = await inventory_service.create_inventory_item(sample_inventory_item)
        initial_stock = item.current_stock
        
        # Create inbound transaction
        transaction_data = {
            "sku": item.sku,
            "transaction_type": TransactionType.INBOUND,
            "quantity": 50,
            "location": item.location,
            "created_by": "test_user"
        }
        
        transaction = await inventory_service.create_inventory_transaction(transaction_data)
        updated_item = await inventory_service.update_stock_levels(item.sku, transaction, item)
        
        assert updated_item.current_stock == initial_stock + 50
        assert updated_item.available_stock == updated_item.current_stock - updated_item.reserved_stock
    
    @pytest.mark.asyncio
    async def test_update_stock_levels_outbound(self, inventory_service, sample_inventory_item):
        """Test stock level updates for outbound transactions."""
        # Create inventory item
        item = await inventory_service.create_inventory_item(sample_inventory_item)
        initial_stock = item.current_stock
        
        # Create outbound transaction
        transaction_data = {
            "sku": item.sku,
            "transaction_type": TransactionType.OUTBOUND,
            "quantity": -30,  # Negative for outbound
            "location": item.location,
            "created_by": "test_user"
        }
        
        transaction = await inventory_service.create_inventory_transaction(transaction_data)
        updated_item = await inventory_service.update_stock_levels(item.sku, transaction, item)
        
        assert updated_item.current_stock == initial_stock - 30
        assert updated_item.available_stock == updated_item.current_stock - updated_item.reserved_stock
    
    @pytest.mark.asyncio
    async def test_update_stock_levels_insufficient_inventory(self, inventory_service):
        """Test stock level updates with insufficient inventory."""
        # Create item with low stock
        item_data = {
            "sku": "LOW-STOCK-001",
            "description": "Low stock item",
            "current_stock": 10,
            "location": "WAREHOUSE-A",
            "unit_cost": 15.00,
            "reorder_point": 20,
            "safety_stock": 15
        }
        
        item = await inventory_service.create_inventory_item(item_data)
        
        # Try to create outbound transaction for more than available
        transaction_data = {
            "sku": item.sku,
            "transaction_type": TransactionType.OUTBOUND,
            "quantity": -50,  # More than available
            "location": item.location,
            "created_by": "test_user"
        }
        
        transaction = await inventory_service.create_inventory_transaction(transaction_data)
        
        with pytest.raises(InsufficientInventoryException):
            await inventory_service.update_stock_levels(item.sku, transaction, item)
    
    @pytest.mark.asyncio
    async def test_export_inventory_recommendations_csv(self, inventory_service, sample_inventory_csv):
        """Test inventory recommendations CSV export."""
        recommendations = await inventory_service.process_inventory_csv(sample_inventory_csv)
        csv_output = await inventory_service.export_inventory_recommendations_csv(recommendations)
        
        assert isinstance(csv_output, str)
        assert "sku,safety_stock,reorder_point,current_stock" in csv_output
        assert "WIDGET-001" in csv_output
        
        # Verify CSV structure
        lines = csv_output.strip().split('\n')
        assert len(lines) == 6  # Header + 5 data rows
    
    @pytest.mark.asyncio
    async def test_get_optimization_summary(self, inventory_service, sample_inventory_csv):
        """Test optimization summary generation."""
        recommendations = await inventory_service.process_inventory_csv(sample_inventory_csv)
        summary = await inventory_service.get_optimization_summary(recommendations.recommendations)
        
        assert summary["total_items"] == 5
        assert "action_breakdown" in summary
        assert "items_requiring_reorder" in summary
        assert "average_confidence_score" in summary
        assert "total_safety_stock_units" in summary
        assert summary["average_confidence_score"] >= 0
        assert summary["average_confidence_score"] <= 1


# Convenience Function Tests

class TestConvenienceFunctions:
    """Test class for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_process_inventory_csv_file(self, sample_inventory_csv):
        """Test inventory CSV file processing convenience function."""
        csv_output, summary = await process_inventory_csv_file(sample_inventory_csv)
        
        assert isinstance(csv_output, str)
        assert isinstance(summary, dict)
        assert summary["mode"] == "inventory_optimization"
        assert summary["total_recommendations"] == 5
        assert "generated_at" in summary
        assert "items_requiring_reorder" in summary


# Data Model Tests

class TestDataModels:
    """Test class for data model validation."""
    
    def test_inventory_status_input_validation(self):
        """Test inventory status input model validation."""
        # Valid data
        valid_data = {
            "sku": "WIDGET-001",
            "current_stock": 150,
            "lead_time_days": 7,
            "service_level_target": 0.95,
            "avg_weekly_demand": 50.0,
            "demand_std_dev": 12.5
        }
        
        item = InventoryStatusInput(**valid_data)
        assert item.sku == "WIDGET-001"
        assert item.current_stock == 150
        assert item.service_level_target == 0.95
    
    def test_inventory_status_input_validation_errors(self):
        """Test inventory status input validation errors."""
        # Invalid service level (too high)
        with pytest.raises(ValueError):
            InventoryStatusInput(
                sku="TEST",
                current_stock=100,
                lead_time_days=7,
                service_level_target=1.5,  # Invalid
                avg_weekly_demand=50.0,
                demand_std_dev=12.5
            )
        
        # Negative stock
        with pytest.raises(ValueError):
            InventoryStatusInput(
                sku="TEST",
                current_stock=-10,
                lead_time_days=7,
                service_level_target=0.95,
                avg_weekly_demand=50.0,
                demand_std_dev=12.5
            )
    
    def test_inventory_item_validation(self, sample_inventory_item):
        """Test inventory item model validation."""
        from inventory.models import InventoryItem
        
        # Add calculated field
        sample_inventory_item["available_stock"] = sample_inventory_item["current_stock"]
        
        item = InventoryItem(**sample_inventory_item)
        assert item.sku == sample_inventory_item["sku"]
        assert item.current_stock == sample_inventory_item["current_stock"]
        assert item.available_stock == sample_inventory_item["current_stock"]
    
    def test_inventory_transaction_validation(self, sample_transaction):
        """Test inventory transaction model validation."""
        from inventory.models import InventoryTransaction
        
        # Add required fields
        sample_transaction["transaction_id"] = "TXN-TEST-001"
        
        transaction = InventoryTransaction(**sample_transaction)
        assert transaction.sku == sample_transaction["sku"]
        assert transaction.transaction_type == sample_transaction["transaction_type"]
        assert transaction.quantity == sample_transaction["quantity"]


# Error Handling Tests

class TestErrorHandling:
    """Test class for error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_csv_error(self, inventory_service):
        """Test handling of empty CSV data."""
        with pytest.raises(CSVProcessingError):
            await inventory_service.process_inventory_csv("")
    
    @pytest.mark.asyncio
    async def test_malformed_csv_error(self, inventory_service):
        """Test handling of malformed CSV data."""
        malformed_csv = "sku,stock\nWIDGET-001,not_a_number,extra_column"
        
        with pytest.raises((CSVProcessingError, DataValidationError)):
            await inventory_service.process_inventory_csv(malformed_csv)
    
    @pytest.mark.asyncio
    async def test_duplicate_skus_error(self, inventory_service):
        """Test handling of duplicate SKUs in CSV."""
        duplicate_csv = """sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev
WIDGET-001,150,7,0.95,50.0,12.5
WIDGET-001,100,14,0.90,40.0,10.0"""
        
        with pytest.raises(DataValidationError):
            await inventory_service.process_inventory_csv(duplicate_csv)
    
    @pytest.mark.asyncio
    async def test_invalid_service_level_error(self, inventory_service):
        """Test handling of invalid service level values."""
        invalid_csv = """sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev
WIDGET-001,150,7,0.3,50.0,12.5"""  # Service level too low
        
        with pytest.raises(DataValidationError):
            await inventory_service.process_inventory_csv(invalid_csv)


# Performance Tests

class TestPerformance:
    """Test class for performance validation."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_processing(self, inventory_service):
        """Test processing of larger datasets."""
        # Create larger CSV dataset
        csv_lines = ["sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev"]
        
        for i in range(100):
            csv_lines.append(f"ITEM-{i:03d},{i*10 + 50},{7 + i%14},0.95,{20.0 + i*0.5},{5.0 + i*0.1}")
        
        large_csv = '\n'.join(csv_lines)
        
        start_time = datetime.now()
        result = await inventory_service.process_inventory_csv(large_csv)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert result.total_recommendations == 100
        assert processing_time < 10.0  # Should process 100 items in under 10 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, inventory_service, sample_inventory_csv):
        """Test concurrent processing of multiple requests."""
        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            task = inventory_service.process_inventory_csv(sample_inventory_csv)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        for result in results:
            assert result.total_recommendations == 5
            assert len(result.recommendations) == 5


# Safety Stock Calculation Tests

class TestSafetyStockCalculations:
    """Test class for detailed safety stock calculation scenarios."""
    
    @pytest.mark.asyncio
    async def test_z_score_calculation(self, inventory_service):
        """Test Z-score calculation for different service levels."""
        # Test common service levels
        service_levels = [0.80, 0.90, 0.95, 0.99]
        expected_z_scores = [0.84, 1.28, 1.65, 2.33]
        
        for service_level, expected_z in zip(service_levels, expected_z_scores):
            z_score = inventory_service._get_z_score_for_service_level(service_level)
            assert abs(z_score - expected_z) < 0.1  # Allow small tolerance
    
    @pytest.mark.asyncio
    async def test_demand_variability_impact(self, inventory_service):
        """Test impact of demand variability on safety stock."""
        base_item = {
            "sku": "VAR-TEST",
            "current_stock": 100,
            "lead_time_days": 14,
            "service_level_target": 0.95,
            "avg_weekly_demand": 50.0
        }
        
        # Test low variability
        low_var_item = InventoryStatusInput(**{**base_item, "demand_std_dev": 5.0})
        low_var_result = await inventory_service._calculate_safety_stock(low_var_item)
        
        # Test high variability
        high_var_item = InventoryStatusInput(**{**base_item, "demand_std_dev": 25.0})
        high_var_result = await inventory_service._calculate_safety_stock(high_var_item)
        
        # High variability should result in higher safety stock
        assert high_var_result.calculated_safety_stock > low_var_result.calculated_safety_stock
        assert high_var_result.demand_variability > low_var_result.demand_variability
    
    @pytest.mark.asyncio
    async def test_lead_time_impact(self, inventory_service):
        """Test impact of lead time on safety stock."""
        base_item = {
            "sku": "LT-TEST",
            "current_stock": 100,
            "service_level_target": 0.95,
            "avg_weekly_demand": 50.0,
            "demand_std_dev": 12.5
        }
        
        # Test short lead time
        short_lt_item = InventoryStatusInput(**{**base_item, "lead_time_days": 7})
        short_lt_result = await inventory_service._calculate_safety_stock(short_lt_item)
        
        # Test long lead time
        long_lt_item = InventoryStatusInput(**{**base_item, "lead_time_days": 28})
        long_lt_result = await inventory_service._calculate_safety_stock(long_lt_item)
        
        # Longer lead time should result in higher safety stock
        assert long_lt_result.calculated_safety_stock > short_lt_result.calculated_safety_stock
        assert long_lt_result.reorder_point > short_lt_result.reorder_point


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])