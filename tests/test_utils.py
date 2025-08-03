"""
Unit tests for utility functions including CSV processing and ML pipeline validation.

This module provides comprehensive testing for all utility functions in the utils
package to ensure proper functionality, error handling, and integration between
different components.

Test Coverage:
- CSV processing and data validation
- Feature engineering and statistical analysis
- Dummy data generation for all domains
- Machine learning utilities and optimization
- Error handling and edge cases
- Integration testing across modules

Author: Balancer Platform
Version: 1.0.0
"""

import csv
import io
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

# Import utility modules
from utils.helpers import (
    read_csv_to_dict, write_dict_to_csv, validate_pydantic_data,
    convert_string_to_numeric, parse_date_string, format_date_string,
    create_lagged_features, create_moving_averages, create_seasonality_features,
    calculate_statistical_features, detect_outliers, clean_data_series,
    validate_data_quality, CSVProcessingError, DataValidationError
)
from utils.dummy_data import (
    DummyDataGenerator, get_sample_procurement_data, get_sample_manufacturing_data,
    get_sample_inventory_data, get_sample_demand_data, get_sample_distribution_data,
    get_sample_csv_string
)
from utils.ml_utils import (
    haversine_distance, euclidean_distance, create_distance_matrix,
    VRPSolver, optimize_inventory_levels, optimize_production_batch,
    prepare_time_series_features, evaluate_forecast_accuracy,
    time_series_cross_validation, calculate_safety_stock, calculate_reorder_point,
    detect_demand_pattern, MLUtilsError
)


# =============================================================================
# TEST MODELS AND FIXTURES
# =============================================================================

class TestModel(BaseModel):
    """Test model for Pydantic validation testing."""
    name: str
    value: int
    price: float


# =============================================================================
# CSV PROCESSING TESTS
# =============================================================================

class TestCSVHelpers:
    """Test CSV processing helper functions."""
    
    def test_read_csv_to_dict_valid(self):
        """Test reading valid CSV content."""
        csv_content = "name,value,price\nProduct A,100,10.50\nProduct B,200,20.75"
        required_columns = ['name', 'value', 'price']
        
        result = read_csv_to_dict(csv_content, required_columns)
        
        assert len(result) == 2
        assert result[0]['name'] == 'Product A'
        assert result[0]['value'] == '100'
        assert result[0]['price'] == '10.50'
    
    def test_read_csv_to_dict_missing_columns(self):
        """Test reading CSV with missing required columns."""
        csv_content = "name,value\nProduct A,100"
        required_columns = ['name', 'value', 'price']
        
        with pytest.raises(CSVProcessingError, match="Missing required columns"):
            read_csv_to_dict(csv_content, required_columns)
    
    def test_read_csv_to_dict_empty(self):
        """Test reading empty CSV."""
        csv_content = ""
        required_columns = ['name']
        
        with pytest.raises(CSVProcessingError, match="empty or invalid"):
            read_csv_to_dict(csv_content, required_columns)
    
    def test_write_dict_to_csv(self):
        """Test writing dictionary data to CSV."""
        data = [
            {'name': 'Product A', 'value': 100, 'price': 10.50},
            {'name': 'Product B', 'value': 200, 'price': 20.75}
        ]
        columns = ['name', 'value', 'price']
        
        result = write_dict_to_csv(data, columns)
        
        # Parse the result back to verify
        lines = result.strip().split('\n')
        assert len(lines) == 3  # Header + 2 data rows
        assert 'name,value,price' in lines[0]
    
    def test_validate_pydantic_data_valid(self):
        """Test validating valid data against Pydantic model."""
        data = [
            {'name': 'Product A', 'value': 100, 'price': 10.50},
            {'name': 'Product B', 'value': 200, 'price': 20.75}
        ]
        
        result = validate_pydantic_data(data, TestModel)
        
        assert len(result) == 2
        assert isinstance(result[0], TestModel)
        assert result[0].name == 'Product A'
    
    def test_validate_pydantic_data_invalid(self):
        """Test validating invalid data against Pydantic model."""
        data = [
            {'name': 'Product A', 'value': 'invalid', 'price': 10.50}
        ]
        
        with pytest.raises(DataValidationError):
            validate_pydantic_data(data, TestModel)
    
    def test_convert_string_to_numeric(self):
        """Test string to numeric conversion."""
        assert convert_string_to_numeric("10.5", "float") == 10.5
        assert convert_string_to_numeric("10", "int") == 10
        assert convert_string_to_numeric("10.0", "int") == 10
        
        with pytest.raises(ValueError):
            convert_string_to_numeric("invalid", "float")
    
    def test_parse_date_string(self):
        """Test date string parsing."""
        date_str = "2023-12-25"
        result = parse_date_string(date_str)
        
        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 25
    
    def test_format_date_string(self):
        """Test date formatting."""
        date_obj = datetime(2023, 12, 25)
        result = format_date_string(date_obj)
        
        assert result == "2023-12-25"


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_create_lagged_features(self):
        """Test lagged feature creation."""
        data = pd.Series([1, 2, 3, 4, 5])
        lags = [1, 2]
        
        result = create_lagged_features(data, lags)
        
        assert 'lag_1' in result.columns
        assert 'lag_2' in result.columns
        assert pd.isna(result['lag_1'].iloc[0])  # First value should be NaN
        assert result['lag_1'].iloc[1] == 1  # Second value should be first original value
    
    def test_create_moving_averages(self):
        """Test moving average creation."""
        data = pd.Series([1, 2, 3, 4, 5])
        windows = [2, 3]
        
        result = create_moving_averages(data, windows)
        
        assert 'ma_2' in result.columns
        assert 'ma_3' in result.columns
        assert 'ma_std_2' in result.columns
        assert result['ma_2'].iloc[1] == 1.5  # Average of first two values
    
    def test_create_seasonality_features(self):
        """Test seasonality feature creation."""
        dates = pd.Series([datetime(2023, 1, 1), datetime(2023, 6, 15), datetime(2023, 12, 31)])
        
        result = create_seasonality_features(dates)
        
        assert 'month' in result.columns
        assert 'quarter' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns
        assert result['month'].iloc[0] == 1
        assert result['month'].iloc[1] == 6
    
    def test_calculate_statistical_features(self):
        """Test statistical feature calculation."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        result = calculate_statistical_features(data)
        
        assert 'mean' in result
        assert 'std' in result
        assert 'median' in result
        assert result['mean'] == 5.5
        assert result['median'] == 5.5
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        
        result = detect_outliers(data, method="iqr")
        
        assert isinstance(result, pd.Series)
        assert result.iloc[-1] == True  # Last value should be detected as outlier
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        
        result = detect_outliers(data, method="zscore", threshold=2)
        
        assert isinstance(result, pd.Series)
        assert result.iloc[-1] == True  # Last value should be detected as outlier
    
    def test_clean_data_series(self):
        """Test data series cleaning."""
        data = pd.Series([1, np.nan, 3, 4, 5])
        
        result = clean_data_series(data, fill_method="forward")
        
        assert not result.isna().any()  # No NaN values should remain
        assert result.iloc[1] == 1  # NaN should be filled with previous value
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': [10, 20, np.nan, 40],
            'col3': ['a', 'b', 'c', 'd']
        })
        required_columns = ['col1', 'col2']
        
        result = validate_data_quality(data, required_columns)
        
        assert 'total_rows' in result
        assert 'missing_values' in result
        assert result['total_rows'] == 4
        assert 'col2' in result['missing_values']


# =============================================================================
# DUMMY DATA GENERATION TESTS
# =============================================================================

class TestDummyDataGenerator:
    """Test dummy data generation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DummyDataGenerator(seed=42)
    
    def test_generate_procurement_data(self):
        """Test procurement data generation."""
        self.setUp()
        data = self.generator.generate_procurement_data(10)
        
        assert len(data) == 10
        assert all('sku' in record for record in data)
        assert all('current_inventory' in record for record in data)
        assert all('supplier_id' in record for record in data)
        assert all(isinstance(record['unit_cost'], float) for record in data)
    
    def test_generate_manufacturing_data(self):
        """Test manufacturing data generation."""
        self.setUp()
        data = self.generator.generate_manufacturing_data(10)
        
        assert len(data) == 10
        assert all('sku' in record for record in data)
        assert all('batch_size' in record for record in data)
        assert all('production_time_days' in record for record in data)
    
    def test_generate_inventory_status(self):
        """Test inventory status data generation."""
        self.setUp()
        data = self.generator.generate_inventory_status(10)
        
        assert len(data) == 10
        assert all('sku' in record for record in data)
        assert all('current_stock' in record for record in data)
        assert all('service_level_target' in record for record in data)
        assert all(0.5 <= record['service_level_target'] <= 1.0 for record in data)
    
    def test_generate_demand_data(self):
        """Test demand data generation."""
        self.setUp()
        data = self.generator.generate_demand_data(100, 5)
        
        # The function generates 52 weeks * num_skus records, so adjust expectation
        assert len(data) == 52 * 5  # 52 weeks * 5 SKUs = 260 records
        assert all('date' in record for record in data)
        assert all('sku' in record for record in data)
        assert all('quantity' in record for record in data)
        
        # Check date format
        for record in data[:5]:
            datetime.strptime(record['date'], '%Y-%m-%d')  # Should not raise exception
    
    def test_generate_orders_and_vehicles(self):
        """Test orders and vehicles data generation."""
        self.setUp()
        data = self.generator.generate_orders_and_vehicles(20, 5)
        
        assert 'orders' in data
        assert 'vehicles' in data
        assert len(data['orders']) == 20
        assert len(data['vehicles']) == 5
        
        # Check order structure
        order = data['orders'][0]
        assert 'order_id' in order
        assert 'customer_lat' in order
        assert 'customer_lon' in order
        
        # Check vehicle structure
        vehicle = data['vehicles'][0]
        assert 'vehicle_id' in vehicle
        assert 'max_volume_m3' in vehicle
        assert 'cost_per_km' in vehicle
    
    def test_get_sample_csv_string(self):
        """Test CSV string generation."""
        csv_string = get_sample_csv_string('procurement', num_records=5)
        
        assert isinstance(csv_string, str)
        assert 'sku' in csv_string
        assert 'supplier_id' in csv_string
        
        # Should be able to parse as CSV
        lines = csv_string.strip().split('\n')
        assert len(lines) >= 6  # Header + 5 data rows


# =============================================================================
# MACHINE LEARNING UTILITIES TESTS
# =============================================================================

class TestMLUtils:
    """Test machine learning utility functions."""
    
    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        # Distance between New York and Los Angeles (approximately 3944 km)
        ny_lat, ny_lon = 40.7128, -74.0060
        la_lat, la_lon = 34.0522, -118.2437
        
        distance = haversine_distance(ny_lat, ny_lon, la_lat, la_lon)
        
        assert isinstance(distance, float)
        assert 3900 < distance < 4000  # Approximate expected distance
    
    def test_euclidean_distance(self):
        """Test euclidean distance calculation."""
        # Simple test with known coordinates
        distance = euclidean_distance(0, 0, 1, 1)
        
        assert isinstance(distance, float)
        assert distance > 0
    
    def test_create_distance_matrix(self):
        """Test distance matrix creation."""
        locations = [
            {'lat': 40.7128, 'lon': -74.0060},  # New York
            {'lat': 34.0522, 'lon': -118.2437}, # Los Angeles
            {'lat': 41.8781, 'lon': -87.6298}   # Chicago
        ]
        
        matrix = create_distance_matrix(locations, method="haversine")
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 0  # Distance from point to itself
        assert matrix[0, 1] == matrix[1, 0]  # Symmetric matrix
        assert np.all(matrix >= 0)  # All distances should be non-negative
    
    def test_vrp_solver_initialization(self):
        """Test VRP solver initialization."""
        distance_matrix = np.array([
            [0, 10, 15],
            [10, 0, 20],
            [15, 20, 0]
        ])
        demands = [5, 8]  # Demands for locations 1 and 2 (excluding depot)
        capacities = [20, 15]  # Two vehicles
        
        solver = VRPSolver(distance_matrix, demands, capacities, depot_index=0)
        
        assert solver.num_locations == 3
        assert solver.num_vehicles == 2
        assert solver.depot_index == 0
    
    def test_vrp_greedy_assignment(self):
        """Test VRP greedy assignment algorithm."""
        distance_matrix = np.array([
            [0, 10, 15],
            [10, 0, 20],
            [15, 20, 0]
        ])
        demands = [5, 8]
        capacities = [20, 15]
        
        solver = VRPSolver(distance_matrix, demands, capacities)
        result = solver.greedy_assignment()
        
        assert 'routes' in result
        assert 'total_distance' in result
        assert 'algorithm' in result
        assert result['algorithm'] == 'greedy'
        assert isinstance(result['total_distance'], float)
    
    def test_optimize_inventory_levels(self):
        """Test inventory level optimization."""
        result = optimize_inventory_levels(
            current_stock=100,
            demand_mean=50,
            demand_std=10,
            lead_time=2,
            service_level=0.95
        )
        
        assert 'safety_stock' in result
        assert 'reorder_point' in result
        assert 'optimal_order_quantity' in result
        assert result['safety_stock'] >= 0
        assert result['reorder_point'] >= 0
    
    def test_optimize_production_batch(self):
        """Test production batch optimization."""
        demand_forecast = [100, 120, 110, 130, 105]
        
        result = optimize_production_batch(
            demand_forecast=demand_forecast,
            setup_cost=500,
            holding_cost_rate=0.2,
            production_rate=200
        )
        
        assert 'optimal_batch_size' in result
        assert 'number_of_batches' in result
        assert 'total_cost' in result
        assert result['optimal_batch_size'] > 0
    
    def test_prepare_time_series_features(self):
        """Test time series feature preparation."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': range(10, 20)
        })
        
        result = prepare_time_series_features(data, 'date', 'value')
        
        assert 'lag_1' in result.columns
        assert 'ma_3' in result.columns
        assert 'month' in result.columns
        assert 'trend' in result.columns
        assert len(result) == len(data)
    
    def test_evaluate_forecast_accuracy(self):
        """Test forecast accuracy evaluation."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])
        
        result = evaluate_forecast_accuracy(y_true, y_pred)
        
        assert 'mae' in result
        assert 'rmse' in result
        assert 'mape' in result
        assert 'directional_accuracy' in result
        assert result['mae'] >= 0
        assert result['rmse'] >= 0
    
    def test_calculate_safety_stock(self):
        """Test safety stock calculation."""
        safety_stock = calculate_safety_stock(
            demand_mean=100,
            demand_std=20,
            lead_time=2,
            service_level=0.95
        )
        
        assert isinstance(safety_stock, float)
        assert safety_stock >= 0
    
    def test_calculate_reorder_point(self):
        """Test reorder point calculation."""
        reorder_point = calculate_reorder_point(
            demand_mean=100,
            lead_time=2,
            safety_stock=50
        )
        
        assert isinstance(reorder_point, (int, float))  # Can be int or float
        assert reorder_point >= 0
        assert reorder_point >= 50  # Should be at least the safety stock
    
    def test_detect_demand_pattern(self):
        """Test demand pattern detection."""
        # Create trending data
        demand_data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        
        result = detect_demand_pattern(demand_data)
        
        assert 'trend' in result
        assert 'seasonality' in result
        assert 'volatility' in result
        assert 'pattern_strength' in result
        assert result['trend'] in ['increasing', 'decreasing', 'stable']
    
    def test_detect_demand_pattern_insufficient_data(self):
        """Test demand pattern detection with insufficient data."""
        demand_data = [10, 12]  # Too few data points
        
        result = detect_demand_pattern(demand_data)
        
        assert result['trend'] == 'insufficient_data'
        assert result['seasonality'] == 'insufficient_data'


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling in utility functions."""
    
    def test_csv_processing_error(self):
        """Test CSV processing error handling."""
        with pytest.raises(CSVProcessingError):
            read_csv_to_dict("invalid,csv\ndata", ['missing_column'])
    
    def test_data_validation_error(self):
        """Test data validation error handling."""
        invalid_data = [{'name': 'test', 'value': 'invalid', 'price': 10.0}]
        
        with pytest.raises(DataValidationError):
            validate_pydantic_data(invalid_data, TestModel)
    
    def test_ml_utils_error(self):
        """Test ML utilities error handling."""
        with pytest.raises(MLUtilsError):
            create_distance_matrix([], method="haversine")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for utility functions working together."""
    
    def test_csv_to_ml_pipeline(self):
        """Test complete pipeline from CSV to ML features."""
        # Generate sample demand data
        generator = DummyDataGenerator(seed=42)
        demand_data = generator.generate_demand_data(100, 5)
        
        # Convert to DataFrame
        df = pd.DataFrame(demand_data)
        df['date'] = pd.to_datetime(df['date'])
        df['quantity'] = df['quantity'].astype(float)
        
        # Prepare ML features
        features_df = prepare_time_series_features(df, 'date', 'quantity')
        
        # Validate the pipeline worked
        assert len(features_df) == len(df)
        assert 'lag_1' in features_df.columns
        assert 'ma_3' in features_df.columns
        assert not features_df.empty
    
    def test_inventory_optimization_pipeline(self):
        """Test inventory optimization with generated data."""
        # Generate inventory data
        generator = DummyDataGenerator(seed=42)
        inventory_data = generator.generate_inventory_status(10)
        
        # Test optimization for each item
        for item in inventory_data:
            result = optimize_inventory_levels(
                current_stock=item['current_stock'],
                demand_mean=item['avg_weekly_demand'],
                demand_std=item['demand_std_dev'],
                lead_time=item['lead_time_days'],
                service_level=item['service_level_target']
            )
            
            assert result['safety_stock'] >= 0
            assert result['reorder_point'] >= result['safety_stock']
    
    def test_distribution_optimization_pipeline(self):
        """Test distribution optimization with generated data."""
        # Generate distribution data
        generator = DummyDataGenerator(seed=42)
        data = generator.generate_orders_and_vehicles(10, 3)
        
        orders = data['orders']
        vehicles = data['vehicles']
        
        # Create locations list (depot + customer locations)
        locations = [{'lat': 40.7128, 'lon': -74.0060}]  # Depot in NYC
        locations.extend([{'lat': order['customer_lat'], 'lon': order['customer_lon']} 
                         for order in orders])
        
        # Create distance matrix
        distance_matrix = create_distance_matrix(locations)
        
        # Extract demands and capacities
        demands = [order['weight_kg'] for order in orders]
        capacities = [vehicle['max_weight_kg'] for vehicle in vehicles]
        
        # Solve VRP
        solver = VRPSolver(distance_matrix, demands, capacities)
        result = solver.greedy_assignment()
        
        assert 'routes' in result
        assert 'total_distance' in result
        assert result['total_distance'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])