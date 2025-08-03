"""
Unit tests for service layer functions.

This module tests the new DataFrame-based service layer to ensure
proper functionality and integration between different components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Import service classes
from supply.service import SupplyService
from inventory.service import InventoryService
from demand.service import DemandService
from distribution.service import DistributionService

# Import utility modules
from utils.dummy_data import DummyDataGenerator


class TestSupplyService:
    """Test supply service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = SupplyService()
        self.generator = DummyDataGenerator(seed=42)
    
    def test_optimize_procurement(self):
        """Test procurement optimization with DataFrame input."""
        # Generate test data
        procurement_data = self.generator.generate_procurement_data(5)
        df = pd.DataFrame(procurement_data)
        
        # Test optimization
        result = self.service.optimize_procurement(df)
        
        assert result['success'] is True
        assert 'recommendations' in result
        assert isinstance(result['recommendations'], pd.DataFrame)
        assert result['recommendations_count'] > 0
        assert 'processing_time_seconds' in result
        assert 'optimization_summary' in result
    
    def test_optimize_manufacturing(self):
        """Test manufacturing optimization with DataFrame input."""
        # Generate test data
        manufacturing_data = self.generator.generate_manufacturing_data(5)
        df = pd.DataFrame(manufacturing_data)
        
        # Test optimization
        result = self.service.optimize_manufacturing(df)
        
        assert result['success'] is True
        assert 'recommendations' in result
        assert isinstance(result['recommendations'], pd.DataFrame)
        assert result['recommendations_count'] >= 0  # May be 0 if no production needed
        assert 'processing_time_seconds' in result
    
    def test_compare_suppliers(self):
        """Test supplier comparison functionality."""
        # Create test supplier data
        supplier_data = [
            {'sku': 'SKU-001', 'supplier_id': 'SUP-A', 'unit_cost': 20.00, 'lead_time_days': 7, 'quality_rating': 4.5},
            {'sku': 'SKU-001', 'supplier_id': 'SUP-B', 'unit_cost': 18.50, 'lead_time_days': 10, 'quality_rating': 4.2},
            {'sku': 'SKU-002', 'supplier_id': 'SUP-A', 'unit_cost': 15.00, 'lead_time_days': 7, 'quality_rating': 4.5}
        ]
        df = pd.DataFrame(supplier_data)
        
        # Test comparison
        result = self.service.compare_suppliers(df)
        
        assert result['success'] is True
        assert 'comparison_results' in result
        assert isinstance(result['comparison_results'], pd.DataFrame)
        assert 'recommendations' in result
    
    def test_data_format_conversion(self):
        """Test CSV and JSON conversion utilities."""
        # Generate test data
        procurement_data = self.generator.generate_procurement_data(3)
        df = pd.DataFrame(procurement_data)
        
        # Test CSV conversion
        csv_data = self.service.to_csv(df)
        assert isinstance(csv_data, str)
        assert len(csv_data) > 0
        
        # Test CSV import
        imported_df = self.service.from_csv(csv_data)
        assert isinstance(imported_df, pd.DataFrame)
        assert len(imported_df) == len(df)
        
        # Test JSON conversion
        json_data = self.service.to_json(df)
        assert isinstance(json_data, list)
        assert len(json_data) == len(df)
        
        # Test JSON import
        json_df = self.service.from_json(json_data)
        assert isinstance(json_df, pd.DataFrame)
        assert len(json_df) == len(df)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid input data."""
        # Test with missing columns
        invalid_df = pd.DataFrame({'sku': ['SKU-001'], 'invalid_column': [100]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.service.optimize_procurement(invalid_df)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            self.service.optimize_procurement(empty_df)


class TestInventoryService:
    """Test inventory service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = InventoryService()
        self.generator = DummyDataGenerator(seed=42)
    
    def test_calculate_safety_stock(self):
        """Test safety stock calculation with DataFrame input."""
        # Generate test data
        inventory_data = self.generator.generate_inventory_status(5)
        df = pd.DataFrame(inventory_data)
        
        # Test calculation
        result = self.service.calculate_safety_stock(df)
        
        assert result['success'] is True
        assert 'recommendations' in result
        assert isinstance(result['recommendations'], pd.DataFrame)
        assert result['recommendations_count'] > 0
        assert 'optimization_summary' in result
        
        # Check recommendation structure
        recommendations = result['recommendations']
        required_columns = ['sku', 'safety_stock', 'reorder_point', 'recommended_action']
        for col in required_columns:
            assert col in recommendations.columns
    
    def test_analyze_abc_classification(self):
        """Test ABC classification analysis."""
        # Create test data
        abc_data = [
            {'sku': 'SKU-001', 'annual_usage_value': 50000},
            {'sku': 'SKU-002', 'annual_usage_value': 30000},
            {'sku': 'SKU-003', 'annual_usage_value': 10000},
            {'sku': 'SKU-004', 'annual_usage_value': 5000}
        ]
        df = pd.DataFrame(abc_data)
        
        # Test analysis
        result = self.service.analyze_abc_classification(df)
        
        assert result['success'] is True
        assert 'classification_results' in result
        assert 'summary' in result
        assert result['summary']['total_items'] == 4
    
    def test_generate_stock_alerts(self):
        """Test stock alert generation."""
        # Create test data
        stock_data = [
            {'sku': 'SKU-001', 'current_stock': 45, 'reorder_point': 100, 'safety_stock': 50},
            {'sku': 'SKU-002', 'current_stock': 200, 'reorder_point': 150, 'safety_stock': 75}
        ]
        df = pd.DataFrame(stock_data)
        
        # Test alert generation
        result = self.service.generate_stock_alerts(df)
        
        assert result['success'] is True
        assert 'alerts' in result
        assert 'alert_summary' in result
        assert len(result['alerts']) == 2
    
    def test_calculate_turnover_metrics(self):
        """Test inventory turnover calculation."""
        # Create test data
        turnover_data = [
            {'sku': 'SKU-001', 'annual_demand': 2400, 'avg_inventory_value': 5000},
            {'sku': 'SKU-002', 'annual_demand': 1200, 'avg_inventory_value': 2000}
        ]
        df = pd.DataFrame(turnover_data)
        
        # Test calculation
        result = self.service.calculate_turnover_metrics(df)
        
        assert result['success'] is True
        assert 'turnover_metrics' in result
        assert 'summary_statistics' in result


class TestDemandService:
    """Test demand service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DemandService()
        self.generator = DummyDataGenerator(seed=42)
    
    def test_generate_forecast(self):
        """Test demand forecast generation."""
        # Generate test data with sufficient history
        demand_data = self.generator.generate_demand_data(100, 2)  # 100 records, 2 SKUs
        df = pd.DataFrame(demand_data)
        
        # Test forecast generation
        params = {'forecast_horizon_weeks': 4, 'confidence_level': 0.95}
        result = self.service.generate_forecast(df, params)
        
        assert result['success'] is True
        assert 'forecasts' in result
        assert isinstance(result['forecasts'], pd.DataFrame)
        assert result['forecasts_count'] > 0
        assert 'forecast_summary' in result
        
        # Check forecast structure
        forecasts = result['forecasts']
        required_columns = ['sku', 'forecast_date', 'prediction', 'confidence_interval_lower', 'confidence_interval_upper']
        for col in required_columns:
            assert col in forecasts.columns
    
    def test_analyze_demand_patterns(self):
        """Test demand pattern analysis."""
        # Generate test data
        demand_data = self.generator.generate_demand_data(50, 2)
        df = pd.DataFrame(demand_data)
        
        # Test analysis
        result = self.service.analyze_demand_patterns(df)
        
        assert result['success'] is True
        assert 'analyses' in result
        assert 'summary_statistics' in result
        assert result['analyses_count'] > 0
    
    def test_validate_forecast_accuracy(self):
        """Test forecast accuracy validation."""
        # Generate test data
        demand_data = self.generator.generate_demand_data(30, 1)
        historical_df = pd.DataFrame(demand_data)
        
        # Create mock forecast data
        forecast_df = historical_df.copy()
        forecast_df['prediction'] = forecast_df['quantity'] * np.random.uniform(0.9, 1.1, len(forecast_df))
        forecast_df = forecast_df[['date', 'sku', 'prediction']]
        
        # Test validation
        result = self.service.validate_forecast_accuracy(historical_df, forecast_df)
        
        assert result['success'] is True
        assert 'accuracy_results' in result
        assert 'overall_metrics' in result
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for forecasting."""
        # Create insufficient data (less than 28 records)
        insufficient_data = [
            {'date': '2024-01-01', 'sku': 'SKU-001', 'quantity': 100},
            {'date': '2024-01-02', 'sku': 'SKU-001', 'quantity': 110}
        ]
        df = pd.DataFrame(insufficient_data)
        
        with pytest.raises(ValueError, match="Minimum 28 data points required"):
            self.service.generate_forecast(df)


class TestDistributionService:
    """Test distribution service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DistributionService()
        self.generator = DummyDataGenerator(seed=42)
    
    def test_optimize_routes(self):
        """Test route optimization."""
        # Generate test data
        distribution_data = self.generator.generate_orders_and_vehicles(5, 2)
        orders_df = pd.DataFrame(distribution_data['orders'])
        vehicles_df = pd.DataFrame(distribution_data['vehicles'])
        
        # Test optimization
        result = self.service.optimize_routes(orders_df, vehicles_df)
        
        assert result['success'] is True
        assert 'solution' in result
        assert 'routes' in result
        assert 'assignments' in result
        assert isinstance(result['assignments'], pd.DataFrame)
        
        # Check solution structure
        solution = result['solution']
        assert 'total_distance' in solution
        assert 'total_cost' in solution
        assert 'algorithm_used' in solution
    
    def test_calculate_distance_matrix(self):
        """Test distance matrix calculation."""
        # Create test locations
        locations_data = [
            {'location_id': 'DEPOT', 'latitude': 40.7128, 'longitude': -74.0060},
            {'location_id': 'CUST-001', 'latitude': 40.7589, 'longitude': -73.9851},
            {'location_id': 'CUST-002', 'latitude': 40.6892, 'longitude': -74.0445}
        ]
        df = pd.DataFrame(locations_data)
        
        # Test calculation
        result = self.service.calculate_distance_matrix(df)
        
        assert result['success'] is True
        assert 'distance_matrix' in result
        assert 'calculation_method' in result
        assert result['locations_count'] == 3
    
    def test_analyze_route_performance(self):
        """Test route performance analysis."""
        # Create test route data
        routes_data = [
            {'vehicle_id': 'VEH-001', 'total_distance_km': 45.2, 'total_cost': 67.80, 'estimated_duration_hours': 3.5},
            {'vehicle_id': 'VEH-002', 'total_distance_km': 52.1, 'total_cost': 91.18, 'estimated_duration_hours': 4.2}
        ]
        df = pd.DataFrame(routes_data)
        
        # Test analysis
        result = self.service.analyze_route_performance(df)
        
        assert result['success'] is True
        assert 'performance_metrics' in result
        assert 'route_efficiency' in result
    
    def test_analyze_capacity_utilization(self):
        """Test capacity utilization analysis."""
        # Create test data
        assignments_data = [
            {'vehicle_id': 'VEH-001', 'order_id': 'ORD-001'},
            {'vehicle_id': 'VEH-001', 'order_id': 'ORD-002'}
        ]
        vehicles_data = [
            {'vehicle_id': 'VEH-001', 'max_volume_m3': 25.0, 'max_weight_kg': 1000.0, 'cost_per_km': 1.50}
        ]
        orders_data = [
            {'order_id': 'ORD-001', 'volume_m3': 2.5, 'weight_kg': 50.0},
            {'order_id': 'ORD-002', 'volume_m3': 1.8, 'weight_kg': 35.0}
        ]
        
        assignments_df = pd.DataFrame(assignments_data)
        vehicles_df = pd.DataFrame(vehicles_data)
        orders_df = pd.DataFrame(orders_data)
        
        # Test analysis
        result = self.service.analyze_capacity_utilization(assignments_df, vehicles_df, orders_df)
        
        assert result['success'] is True
        assert 'vehicle_utilization' in result
        assert 'summary_statistics' in result


class TestServiceIntegration:
    """Test integration between different services."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.supply_service = SupplyService()
        self.inventory_service = InventoryService()
        self.demand_service = DemandService()
        self.distribution_service = DistributionService()
        self.generator = DummyDataGenerator(seed=42)
    
    def test_supply_inventory_integration(self):
        """Test integration between supply and inventory services."""
        # Generate inventory data
        inventory_data = self.generator.generate_inventory_status(3)
        inventory_df = pd.DataFrame(inventory_data)
        
        # Calculate safety stock
        inventory_result = self.inventory_service.calculate_safety_stock(inventory_df)
        
        # Generate procurement data
        procurement_data = self.generator.generate_procurement_data(3)
        procurement_df = pd.DataFrame(procurement_data)
        
        # Optimize procurement
        supply_result = self.supply_service.optimize_procurement(procurement_df)
        
        # Both should succeed
        assert inventory_result['success'] is True
        assert supply_result['success'] is True
        
        # Results should be compatible (both have SKU columns)
        inventory_skus = set(inventory_result['recommendations']['sku'])
        supply_skus = set(supply_result['recommendations']['sku'])
        
        # At least some overlap expected (though not required for this test)
        assert len(inventory_skus) > 0
        assert len(supply_skus) > 0
    
    def test_demand_supply_integration(self):
        """Test integration between demand and supply services."""
        # Generate demand data
        demand_data = self.generator.generate_demand_data(50, 2)
        demand_df = pd.DataFrame(demand_data)
        
        # Generate forecast
        forecast_result = self.demand_service.generate_forecast(demand_df)
        
        # Generate supply data
        supply_data = self.generator.generate_procurement_data(2)
        supply_df = pd.DataFrame(supply_data)
        
        # Optimize supply
        supply_result = self.supply_service.optimize_procurement(supply_df)
        
        # Both should succeed
        assert forecast_result['success'] is True
        assert supply_result['success'] is True
        
        # Results should contain relevant data
        assert len(forecast_result['forecasts']) > 0
        assert len(supply_result['recommendations']) > 0
    
    def test_data_format_consistency(self):
        """Test that all services handle data formats consistently."""
        # Test CSV conversion consistency
        test_data = [
            {'sku': 'SKU-001', 'value': 100},
            {'sku': 'SKU-002', 'value': 200}
        ]
        df = pd.DataFrame(test_data)
        
        # Test all services can convert to/from CSV
        supply_csv = self.supply_service.to_csv(df)
        inventory_csv = self.inventory_service.to_csv(df)
        demand_csv = self.demand_service.to_csv(df)
        distribution_csv = self.distribution_service.to_csv(df)
        
        # All should produce valid CSV
        assert isinstance(supply_csv, str) and len(supply_csv) > 0
        assert isinstance(inventory_csv, str) and len(inventory_csv) > 0
        assert isinstance(demand_csv, str) and len(demand_csv) > 0
        assert isinstance(distribution_csv, str) and len(distribution_csv) > 0
        
        # All should be able to import back
        supply_df = self.supply_service.from_csv(supply_csv)
        inventory_df = self.inventory_service.from_csv(inventory_csv)
        demand_df = self.demand_service.from_csv(demand_csv)
        distribution_df = self.distribution_service.from_csv(distribution_csv)
        
        # All should have same structure
        assert len(supply_df) == len(df)
        assert len(inventory_df) == len(df)
        assert len(demand_df) == len(df)
        assert len(distribution_df) == len(df)


if __name__ == "__main__":
    pytest.main([__file__])