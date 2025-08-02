"""
Balancer Platform Utilities Package

This package provides comprehensive utility functions for the Balancer platform,
including CSV processing, data validation, machine learning algorithms, and
dummy data generation for testing and development.

Modules:
- helpers: CSV processing, data validation, and feature engineering
- dummy_data: Realistic data generation for all domain modules  
- ml_utils: Machine learning algorithms and optimization functions

Key Features:
- Robust CSV processing with error handling
- Pydantic model validation
- Time series feature engineering
- Statistical analysis and outlier detection
- Vehicle routing optimization
- Inventory and production optimization
- Comprehensive test data generation

Author: Balancer Platform
Version: 1.0.0
"""

from .helpers import (
    # CSV Processing
    read_csv_to_dict,
    write_dict_to_csv,
    save_csv_file,
    load_csv_file,
    
    # Data Validation
    validate_pydantic_data,
    validate_data_quality,
    
    # Data Type Conversion
    convert_string_to_numeric,
    parse_date_string,
    format_date_string,
    
    # Feature Engineering
    create_lagged_features,
    create_moving_averages,
    create_seasonality_features,
    
    # Statistical Analysis
    calculate_statistical_features,
    detect_outliers,
    clean_data_series,
    
    # Exceptions
    CSVProcessingError,
    DataValidationError,
)

from .dummy_data import (
    # Main Generator Class
    DummyDataGenerator,
    
    # Convenience Functions
    get_sample_procurement_data,
    get_sample_manufacturing_data,
    get_sample_inventory_data,
    get_sample_demand_data,
    get_sample_distribution_data,
    get_sample_csv_string,
    
    # Test Compatibility Functions
    generate_procurement_data,
    generate_manufacturing_data,
    generate_inventory_data,
    generate_demand_data,
    generate_distribution_data,
)

from .ml_utils import (
    # Distance Calculations
    haversine_distance,
    euclidean_distance,
    create_distance_matrix,
    
    # VRP Optimization
    VRPSolver,
    
    # Inventory Optimization
    optimize_inventory_levels,
    optimize_production_batch,
    calculate_safety_stock,
    calculate_reorder_point,
    
    # ML Features and Evaluation
    prepare_time_series_features,
    evaluate_forecast_accuracy,
    time_series_cross_validation,
    
    # Statistical Analysis
    detect_demand_pattern,
    
    # Exceptions
    MLUtilsError,
)

__version__ = "1.0.0"
__author__ = "Balancer Platform"

# Package metadata
__all__ = [
    # helpers module
    "read_csv_to_dict",
    "write_dict_to_csv",
    "save_csv_file",
    "load_csv_file",
    "validate_pydantic_data",
    "validate_data_quality",
    "convert_string_to_numeric",
    "parse_date_string",
    "format_date_string",
    "create_lagged_features",
    "create_moving_averages",
    "create_seasonality_features",
    "calculate_statistical_features",
    "detect_outliers",
    "clean_data_series",
    "CSVProcessingError",
    "DataValidationError",
    
    # dummy_data module
    "DummyDataGenerator",
    "get_sample_procurement_data",
    "get_sample_manufacturing_data",
    "get_sample_inventory_data",
    "get_sample_demand_data",
    "get_sample_distribution_data",
    "get_sample_csv_string",
    "generate_procurement_data",
    "generate_manufacturing_data",
    "generate_inventory_data",
    "generate_demand_data",
    "generate_distribution_data",
    
    # ml_utils module
    "haversine_distance",
    "euclidean_distance",
    "create_distance_matrix",
    "VRPSolver",
    "optimize_inventory_levels",
    "optimize_production_batch",
    "calculate_safety_stock",
    "calculate_reorder_point",
    "prepare_time_series_features",
    "evaluate_forecast_accuracy",
    "time_series_cross_validation",
    "detect_demand_pattern",
    "MLUtilsError",
]