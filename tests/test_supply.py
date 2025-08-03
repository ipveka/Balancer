"""
Supply Module Tests

Comprehensive test suite for the supply management module including
API endpoint tests, service logic tests, and error handling validation.
"""

import pytest
import asyncio
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json
import io

from supply.api import router as supply_router
from supply.service import SupplyService, process_procurement_csv_file, process_manufacturing_csv_file
from supply.models import (
    SupplyOptimizationRequest, SupplyMode, ProcurementDataInput, ManufacturingDataInput,
    RecommendationAction, SupplyChainException, OptimizationFailedException
)
from utils.helpers import CSVProcessingError, DataValidationError


# Test fixtures
@pytest.fixture
def app():
    """Create FastAPI app with supply router for testing."""
    app = FastAPI()
    app.include_router(supply_router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def supply_service():
    """Create supply service instance."""
    return SupplyService()


@pytest.fixture
def sample_procurement_csv():
    """Sample procurement CSV data for testing."""
    return """sku,current_inventory,forecast_demand_4weeks,safety_stock,min_order_qty,supplier_id,unit_cost
WIDGET-001,100,500,50,100,SUP-001,10.50
GADGET-002,25,200,30,50,SUP-002,25.75
TOOL-003,0,150,20,25,SUP-001,15.25"""


@pytest.fixture
def sample_manufacturing_csv():
    """Sample manufacturing CSV data for testing."""
    return """sku,current_inventory,forecast_demand_4weeks,safety_stock,batch_size,production_time_days,unit_cost
WIDGET-001,100,500,50,200,5,8.50
GADGET-002,25,200,30,100,3,18.75
TOOL-003,0,150,20,50,7,12.25"""


@pytest.fixture
def invalid_csv():
    """Invalid CSV data for error testing."""
    return """invalid,headers,format
some,data,here"""


@pytest.fixture
def missing_columns_csv():
    """CSV with missing required columns."""
    return """sku,current_inventory
WIDGET-001,100"""


# API Endpoint Tests

class TestSupplyAPI:
    """Test class for supply API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/supply/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["module"] == "supply"
        assert "supported_modes" in data
        assert "procurement" in data["supported_modes"]
        assert "manufacturing" in data["supported_modes"]
    
    def test_get_supply_modes(self, client):
        """Test get supply modes endpoint."""
        response = client.get("/supply/modes")
        assert response.status_code == 200
        
        data = response.json()
        assert "supported_modes" in data
        assert len(data["supported_modes"]) == 2
        
        modes = [mode["mode"] for mode in data["supported_modes"]]
        assert "procurement" in modes
        assert "manufacturing" in modes
    
    def test_get_supply_status(self, client):
        """Test get supply status endpoint."""
        response = client.get("/supply/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["module"] == "supply"
        assert data["status"] == "operational"
        assert "features" in data
        assert "configuration" in data
    
    def test_download_procurement_template(self, client):
        """Test procurement template download."""
        response = client.get("/supply/templates/procurement")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "procurement_template.csv" in response.headers["content-disposition"]
        
        # Verify CSV content
        content = response.content.decode('utf-8')
        assert "sku,current_inventory,forecast_demand_4weeks" in content
        assert "WIDGET-001" in content
    
    def test_download_manufacturing_template(self, client):
        """Test manufacturing template download."""
        response = client.get("/supply/templates/manufacturing")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "manufacturing_template.csv" in response.headers["content-disposition"]
        
        # Verify CSV content
        content = response.content.decode('utf-8')
        assert "sku,current_inventory,forecast_demand_4weeks" in content
        assert "batch_size,production_time_days" in content
    
    def test_optimize_supply_procurement(self, client, sample_procurement_csv):
        """Test supply optimization endpoint with procurement mode."""
        request_data = {
            "mode": "procurement",
            "csv_data": sample_procurement_csv,
            "optimization_params": {"test_param": "value"}
        }
        
        response = client.post("/supply/optimize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["mode"] == "procurement"
        assert data["recommendations_count"] == 3
        assert "csv_output" in data
        assert "processing_time_seconds" in data
        assert "optimization_summary" in data
    
    def test_optimize_supply_manufacturing(self, client, sample_manufacturing_csv):
        """Test supply optimization endpoint with manufacturing mode."""
        request_data = {
            "mode": "manufacturing",
            "csv_data": sample_manufacturing_csv,
            "optimization_params": {}
        }
        
        response = client.post("/supply/optimize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["mode"] == "manufacturing"
        assert data["recommendations_count"] == 3
        assert "csv_output" in data
    
    def test_optimize_procurement_specific(self, client, sample_procurement_csv):
        """Test procurement-specific optimization endpoint."""
        request_data = {
            "mode": "manufacturing",  # This should be overridden
            "csv_data": sample_procurement_csv,
            "optimization_params": {}
        }
        
        response = client.post("/supply/procurement/optimize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["mode"] == "procurement"  # Should be overridden
    
    def test_optimize_manufacturing_specific(self, client, sample_manufacturing_csv):
        """Test manufacturing-specific optimization endpoint."""
        request_data = {
            "mode": "procurement",  # This should be overridden
            "csv_data": sample_manufacturing_csv,
            "optimization_params": {}
        }
        
        response = client.post("/supply/manufacturing/optimize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["mode"] == "manufacturing"  # Should be overridden
    
    def test_optimize_supply_invalid_csv(self, client, invalid_csv):
        """Test optimization with invalid CSV data."""
        request_data = {
            "mode": "procurement",
            "csv_data": invalid_csv,
            "optimization_params": {}
        }
        
        response = client.post("/supply/optimize", json=request_data)
        assert response.status_code == 400
        assert "CSV processing failed" in response.json()["detail"]
    
    def test_optimize_supply_missing_columns(self, client, missing_columns_csv):
        """Test optimization with missing required columns."""
        request_data = {
            "mode": "procurement",
            "csv_data": missing_columns_csv,
            "optimization_params": {}
        }
        
        response = client.post("/supply/optimize", json=request_data)
        assert response.status_code == 400
        assert "Missing required columns" in response.json()["detail"]
    
    def test_upload_procurement_csv(self, client, sample_procurement_csv):
        """Test procurement CSV file upload."""
        files = {"file": ("test.csv", io.StringIO(sample_procurement_csv), "text/csv")}
        
        response = client.post("/supply/procurement/upload", files=files)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "procurement_recommendations.csv" in response.headers["content-disposition"]
    
    def test_upload_manufacturing_csv(self, client, sample_manufacturing_csv):
        """Test manufacturing CSV file upload."""
        files = {"file": ("test.csv", io.StringIO(sample_manufacturing_csv), "text/csv")}
        
        response = client.post("/supply/manufacturing/upload", files=files)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "manufacturing_recommendations.csv" in response.headers["content-disposition"]
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        files = {"file": ("test.txt", io.StringIO("test content"), "text/plain")}
        
        response = client.post("/supply/procurement/upload", files=files)
        assert response.status_code == 400
        assert "File must be a CSV file" in response.json()["detail"]


# Service Layer Tests

class TestSupplyService:
    """Test class for supply service layer."""
    
    @pytest.mark.asyncio
    async def test_process_procurement_csv_success(self, supply_service, sample_procurement_csv):
        """Test successful procurement CSV processing."""
        result = await supply_service.process_procurement_csv(sample_procurement_csv)
        
        assert result.total_recommendations == 3
        assert len(result.recommendations) == 3
        
        # Check first recommendation
        rec = result.recommendations[0]
        assert rec.sku == "WIDGET-001"
        assert rec.recommended_quantity >= 0
        assert rec.supplier_id == "DEFAULT_SUPPLIER"  # Default since not in optimization factors
        assert rec.total_cost >= 0
        assert rec.confidence_score >= 0 and rec.confidence_score <= 1
    
    @pytest.mark.asyncio
    async def test_process_manufacturing_csv_success(self, supply_service, sample_manufacturing_csv):
        """Test successful manufacturing CSV processing."""
        result = await supply_service.process_manufacturing_csv(sample_manufacturing_csv)
        
        assert result.total_recommendations == 3
        assert len(result.recommendations) == 3
        
        # Check first recommendation
        rec = result.recommendations[0]
        assert rec.sku == "WIDGET-001"
        assert rec.recommended_batch_qty >= 0
        assert rec.total_cost >= 0
        assert rec.confidence_score >= 0 and rec.confidence_score <= 1
    
    @pytest.mark.asyncio
    async def test_process_procurement_csv_invalid_data(self, supply_service, invalid_csv):
        """Test procurement CSV processing with invalid data."""
        with pytest.raises(CSVProcessingError):
            await supply_service.process_procurement_csv(invalid_csv)
    
    @pytest.mark.asyncio
    async def test_process_manufacturing_csv_invalid_data(self, supply_service, invalid_csv):
        """Test manufacturing CSV processing with invalid data."""
        with pytest.raises(CSVProcessingError):
            await supply_service.process_manufacturing_csv(invalid_csv)
    
    @pytest.mark.asyncio
    async def test_optimize_procurement_item(self, supply_service):
        """Test procurement item optimization logic."""
        item = ProcurementDataInput(
            sku="TEST-001",
            current_inventory=100,
            forecast_demand_4weeks=500,
            safety_stock=50,
            min_order_qty=100,
            supplier_id="SUP-001",
            unit_cost=10.0
        )
        
        result = await supply_service._optimize_procurement_item(item)
        
        assert result.sku == "TEST-001"
        assert result.mode == SupplyMode.PROCUREMENT
        assert result.current_inventory == 100
        assert result.demand_forecast == 500
        assert result.safety_stock_requirement == 50
        
        # Should need 450 units (500 + 50 - 100), rounded up to min order qty
        expected_need = 500 + 50 - 100  # 450
        assert result.calculated_need == expected_need
        assert result.optimized_quantity >= expected_need
        assert result.optimized_quantity % 100 == 0  # Should be multiple of min_order_qty
    
    @pytest.mark.asyncio
    async def test_optimize_manufacturing_item(self, supply_service):
        """Test manufacturing item optimization logic."""
        item = ManufacturingDataInput(
            sku="TEST-001",
            current_inventory=100,
            forecast_demand_4weeks=500,
            safety_stock=50,
            batch_size=200,
            production_time_days=5,
            unit_cost=8.0
        )
        
        result = await supply_service._optimize_manufacturing_item(item)
        
        assert result.sku == "TEST-001"
        assert result.mode == SupplyMode.MANUFACTURING
        assert result.current_inventory == 100
        assert result.demand_forecast == 500
        assert result.safety_stock_requirement == 50
        
        # Should need 450 units, rounded up to batch size
        expected_need = 500 + 50 - 100  # 450
        assert result.calculated_need == expected_need
        assert result.optimized_quantity >= expected_need
        assert result.optimized_quantity % 200 == 0  # Should be multiple of batch_size
    
    @pytest.mark.asyncio
    async def test_export_procurement_recommendations_csv(self, supply_service, sample_procurement_csv):
        """Test procurement recommendations CSV export."""
        recommendations = await supply_service.process_procurement_csv(sample_procurement_csv)
        csv_output = await supply_service.export_procurement_recommendations_csv(recommendations)
        
        assert isinstance(csv_output, str)
        assert "sku,recommended_quantity,supplier_id" in csv_output
        assert "WIDGET-001" in csv_output
        
        # Verify CSV structure
        lines = csv_output.strip().split('\n')
        assert len(lines) == 4  # Header + 3 data rows
    
    @pytest.mark.asyncio
    async def test_export_manufacturing_recommendations_csv(self, supply_service, sample_manufacturing_csv):
        """Test manufacturing recommendations CSV export."""
        recommendations = await supply_service.process_manufacturing_csv(sample_manufacturing_csv)
        csv_output = await supply_service.export_manufacturing_recommendations_csv(recommendations)
        
        assert isinstance(csv_output, str)
        assert "sku,recommended_batch_qty,production_start_date" in csv_output
        assert "WIDGET-001" in csv_output
        
        # Verify CSV structure
        lines = csv_output.strip().split('\n')
        assert len(lines) == 4  # Header + 3 data rows


# Convenience Function Tests

class TestConvenienceFunctions:
    """Test class for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_process_procurement_csv_file(self, sample_procurement_csv):
        """Test procurement CSV file processing convenience function."""
        csv_output, summary = await process_procurement_csv_file(sample_procurement_csv)
        
        assert isinstance(csv_output, str)
        assert isinstance(summary, dict)
        assert summary["mode"] == "procurement"
        assert summary["total_recommendations"] == 3
        assert "generated_at" in summary
    
    @pytest.mark.asyncio
    async def test_process_manufacturing_csv_file(self, sample_manufacturing_csv):
        """Test manufacturing CSV file processing convenience function."""
        csv_output, summary = await process_manufacturing_csv_file(sample_manufacturing_csv)
        
        assert isinstance(csv_output, str)
        assert isinstance(summary, dict)
        assert summary["mode"] == "manufacturing"
        assert summary["total_recommendations"] == 3
        assert "generated_at" in summary


# Data Model Tests

class TestDataModels:
    """Test class for data model validation."""
    
    def test_procurement_data_input_validation(self):
        """Test procurement data input model validation."""
        # Valid data
        valid_data = {
            "sku": "WIDGET-001",
            "current_inventory": 100,
            "forecast_demand_4weeks": 500,
            "safety_stock": 50,
            "min_order_qty": 100,
            "supplier_id": "SUP-001",
            "unit_cost": 10.50
        }
        
        item = ProcurementDataInput(**valid_data)
        assert item.sku == "WIDGET-001"
        assert item.current_inventory == 100
        assert item.unit_cost == 10.50
    
    def test_procurement_data_input_validation_errors(self):
        """Test procurement data input validation errors."""
        # Negative inventory
        with pytest.raises(ValueError):
            ProcurementDataInput(
                sku="TEST",
                current_inventory=-10,
                forecast_demand_4weeks=100,
                safety_stock=10,
                min_order_qty=50,
                supplier_id="SUP-001",
                unit_cost=10.0
            )
        
        # Zero unit cost
        with pytest.raises(ValueError):
            ProcurementDataInput(
                sku="TEST",
                current_inventory=100,
                forecast_demand_4weeks=100,
                safety_stock=10,
                min_order_qty=50,
                supplier_id="SUP-001",
                unit_cost=0.0
            )
    
    def test_manufacturing_data_input_validation(self):
        """Test manufacturing data input model validation."""
        # Valid data
        valid_data = {
            "sku": "WIDGET-001",
            "current_inventory": 100,
            "forecast_demand_4weeks": 500,
            "safety_stock": 50,
            "batch_size": 200,
            "production_time_days": 5,
            "unit_cost": 8.50
        }
        
        item = ManufacturingDataInput(**valid_data)
        assert item.sku == "WIDGET-001"
        assert item.batch_size == 200
        assert item.production_time_days == 5
    
    def test_manufacturing_data_input_validation_errors(self):
        """Test manufacturing data input validation errors."""
        # Invalid production time (too long)
        with pytest.raises(ValueError):
            ManufacturingDataInput(
                sku="TEST",
                current_inventory=100,
                forecast_demand_4weeks=100,
                safety_stock=10,
                batch_size=50,
                production_time_days=400,  # Too long
                unit_cost=10.0
            )


# Error Handling Tests

class TestErrorHandling:
    """Test class for error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_csv_error(self, supply_service):
        """Test handling of empty CSV data."""
        with pytest.raises(CSVProcessingError):
            await supply_service.process_procurement_csv("")
    
    @pytest.mark.asyncio
    async def test_malformed_csv_error(self, supply_service):
        """Test handling of malformed CSV data."""
        malformed_csv = "sku,inventory\nWIDGET-001,not_a_number,extra_column"
        
        with pytest.raises((CSVProcessingError, DataValidationError)):
            await supply_service.process_procurement_csv(malformed_csv)
    
    @pytest.mark.asyncio
    async def test_duplicate_skus_error(self, supply_service):
        """Test handling of duplicate SKUs in CSV."""
        duplicate_csv = """sku,current_inventory,forecast_demand_4weeks,safety_stock,min_order_qty,supplier_id,unit_cost
WIDGET-001,100,500,50,100,SUP-001,10.50
WIDGET-001,200,300,40,100,SUP-002,12.00"""
        
        with pytest.raises(DataValidationError):
            await supply_service.process_procurement_csv(duplicate_csv)


# Performance Tests

class TestPerformance:
    """Test class for performance validation."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_processing(self, supply_service):
        """Test processing of larger datasets."""
        # Create larger CSV dataset
        csv_lines = ["sku,current_inventory,forecast_demand_4weeks,safety_stock,min_order_qty,supplier_id,unit_cost"]
        
        for i in range(100):
            csv_lines.append(f"ITEM-{i:03d},{i*10},{i*50},{i*5},{max(10, i*2)},SUP-{i%5:03d},{10.0 + i*0.1}")
        
        large_csv = '\n'.join(csv_lines)
        
        start_time = datetime.now()
        result = await supply_service.process_procurement_csv(large_csv)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert result.total_recommendations == 100
        assert processing_time < 5.0  # Should process 100 items in under 5 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, supply_service, sample_procurement_csv):
        """Test concurrent processing of multiple requests."""
        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            task = supply_service.process_procurement_csv(sample_procurement_csv)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        for result in results:
            assert result.total_recommendations == 3
            assert len(result.recommendations) == 3


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])