"""
Helper utilities for CSV processing, data validation, and ML feature engineering.

This module provides common utility functions used across all domain modules
for data processing, validation, and machine learning feature engineering.

Key Features:
- CSV processing with robust error handling
- Pydantic model validation
- Time series feature engineering
- Statistical analysis and outlier detection
- Data quality validation and cleaning
- File I/O operations with proper error handling

Author: Balancer Platform
Version: 1.0.0
"""

import csv
import io
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class CSVProcessingError(Exception):
    """
    Custom exception for CSV processing errors.
    
    Raised when CSV reading, writing, or parsing operations fail.
    """
    pass


class DataValidationError(Exception):
    """
    Custom exception for data validation errors.
    
    Raised when data validation against Pydantic models or 
    quality checks fail.
    """
    pass


# =============================================================================
# CSV PROCESSING FUNCTIONS
# =============================================================================


def read_csv_to_dict(csv_content: str, required_columns: List[str]) -> List[Dict[str, Any]]:
    """
    Read CSV content and convert to list of dictionaries with validation.
    
    This function parses CSV content, validates required columns are present,
    and returns clean data with empty values filtered out.
    
    Args:
        csv_content (str): Raw CSV content as string
        required_columns (List[str]): List of column names that must be present
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries representing CSV rows
        
    Raises:
        CSVProcessingError: If CSV is invalid, empty, or missing required columns
        
    Example:
        >>> csv_data = "name,value\\nProduct A,100\\nProduct B,200"
        >>> result = read_csv_to_dict(csv_data, ['name', 'value'])
        >>> len(result)
        2
    """
    try:
        # Use StringIO to read CSV from string
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        
        # Check if all required columns are present
        if not reader.fieldnames:
            raise CSVProcessingError("CSV file appears to be empty or invalid")
            
        missing_columns = set(required_columns) - set(reader.fieldnames)
        if missing_columns:
            raise CSVProcessingError(f"Missing required columns: {missing_columns}")
        
        # Convert to list of dictionaries
        data = []
        for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
            # Remove empty values and strip whitespace
            cleaned_row = {k: v.strip() if isinstance(v, str) else v 
                          for k, v in row.items() if v is not None and v != ''}
            
            if cleaned_row:  # Only add non-empty rows
                data.append(cleaned_row)
        
        if not data:
            raise CSVProcessingError("No valid data rows found in CSV")
            
        return data
        
    except csv.Error as e:
        raise CSVProcessingError(f"CSV parsing error: {str(e)}")
    except Exception as e:
        raise CSVProcessingError(f"Unexpected error reading CSV: {str(e)}")


def write_dict_to_csv(data: List[Dict[str, Any]], columns: List[str]) -> str:
    """
    Write list of dictionaries to CSV string format.
    
    Converts structured data to CSV format with specified column order.
    Missing columns in data are filled with empty strings.
    
    Args:
        data (List[Dict[str, Any]]): List of dictionaries to convert
        columns (List[str]): Column names in desired output order
        
    Returns:
        str: CSV content as string with headers and data rows
        
    Raises:
        CSVProcessingError: If CSV writing operation fails
        
    Example:
        >>> data = [{'name': 'A', 'value': 1}, {'name': 'B', 'value': 2}]
        >>> csv_str = write_dict_to_csv(data, ['name', 'value'])
        >>> 'name,value' in csv_str
        True
    """
    try:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for row in data:
            # Ensure all required columns are present
            filtered_row = {col: row.get(col, '') for col in columns}
            writer.writerow(filtered_row)
        
        return output.getvalue()
        
    except Exception as e:
        raise CSVProcessingError(f"Error writing CSV: {str(e)}")


# =============================================================================
# DATA VALIDATION FUNCTIONS
# =============================================================================

def validate_pydantic_data(data: List[Dict[str, Any]], model_class: BaseModel) -> List[BaseModel]:
    """
    Validate list of dictionaries against Pydantic model.
    
    Args:
        data: List of dictionaries to validate
        model_class: Pydantic model class for validation
        
    Returns:
        List of validated Pydantic model instances
        
    Raises:
        DataValidationError: If validation fails
    """
    validated_data = []
    errors = []
    
    for i, row in enumerate(data):
        try:
            validated_row = model_class(**row)
            validated_data.append(validated_row)
        except ValidationError as e:
            errors.append(f"Row {i + 1}: {str(e)}")
    
    if errors:
        raise DataValidationError(f"Validation errors found:\n" + "\n".join(errors))
    
    return validated_data


# =============================================================================
# DATA TYPE CONVERSION FUNCTIONS
# =============================================================================

def convert_string_to_numeric(value: str, data_type: str = "float") -> Union[int, float]:
    """
    Convert string value to numeric type with error handling.
    
    Args:
        value: String value to convert
        data_type: Target data type ("int" or "float")
        
    Returns:
        Converted numeric value
        
    Raises:
        ValueError: If conversion fails
    """
    try:
        if data_type == "int":
            return int(float(value))  # Handle cases like "10.0"
        elif data_type == "float":
            return float(value)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert '{value}' to {data_type}: {str(e)}")


# =============================================================================
# DATE AND TIME UTILITIES
# =============================================================================

def parse_date_string(date_str: str, date_format: str = "%Y-%m-%d") -> datetime:
    """
    Parse date string to datetime object.
    
    Args:
        date_str: Date string to parse
        date_format: Expected date format
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If date parsing fails
    """
    try:
        return datetime.strptime(date_str, date_format)
    except ValueError as e:
        # Try common alternative formats
        alternative_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]
        
        for fmt in alternative_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Cannot parse date '{date_str}'. Expected format: {date_format}")


def format_date_string(date_obj: datetime, date_format: str = "%Y-%m-%d") -> str:
    """
    Format datetime object to string.
    
    Args:
        date_obj: Datetime object to format
        date_format: Desired date format
        
    Returns:
        Formatted date string
    """
    return date_obj.strftime(date_format)


# =============================================================================
# MACHINE LEARNING FEATURE ENGINEERING
# =============================================================================

def create_lagged_features(data: pd.Series, lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for time series data.
    
    Args:
        data: Time series data as pandas Series
        lags: List of lag periods to create
        
    Returns:
        DataFrame with lagged features
    """
    lagged_df = pd.DataFrame()
    
    for lag in lags:
        lagged_df[f'lag_{lag}'] = data.shift(lag)
    
    return lagged_df


def create_moving_averages(data: pd.Series, windows: List[int]) -> pd.DataFrame:
    """
    Create moving average features for time series data.
    
    Args:
        data: Time series data as pandas Series
        windows: List of window sizes for moving averages
        
    Returns:
        DataFrame with moving average features
    """
    ma_df = pd.DataFrame()
    
    for window in windows:
        ma_df[f'ma_{window}'] = data.rolling(window=window, min_periods=1).mean()
        ma_df[f'ma_std_{window}'] = data.rolling(window=window, min_periods=1).std()
    
    return ma_df


def create_seasonality_features(dates: pd.Series) -> pd.DataFrame:
    """
    Create seasonality features from date series.
    
    Args:
        dates: Date series as pandas Series
        
    Returns:
        DataFrame with seasonality features
    """
    seasonality_df = pd.DataFrame()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)
    
    # Time-based features
    seasonality_df['year'] = dates.dt.year
    seasonality_df['month'] = dates.dt.month
    seasonality_df['quarter'] = dates.dt.quarter
    seasonality_df['week_of_year'] = dates.dt.isocalendar().week
    seasonality_df['day_of_week'] = dates.dt.dayofweek
    seasonality_df['day_of_month'] = dates.dt.day
    seasonality_df['day_of_year'] = dates.dt.dayofyear
    
    # Cyclical encoding for better ML performance
    seasonality_df['month_sin'] = np.sin(2 * np.pi * seasonality_df['month'] / 12)
    seasonality_df['month_cos'] = np.cos(2 * np.pi * seasonality_df['month'] / 12)
    seasonality_df['day_of_week_sin'] = np.sin(2 * np.pi * seasonality_df['day_of_week'] / 7)
    seasonality_df['day_of_week_cos'] = np.cos(2 * np.pi * seasonality_df['day_of_week'] / 7)
    
    return seasonality_df


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def calculate_statistical_features(data: pd.Series, window: int = 4) -> Dict[str, float]:
    """
    Calculate statistical features for a data series.
    
    Args:
        data: Data series for calculation
        window: Window size for rolling statistics
        
    Returns:
        Dictionary of statistical features
    """
    features = {}
    
    # Basic statistics
    features['mean'] = data.mean()
    features['median'] = data.median()
    features['std'] = data.std()
    features['var'] = data.var()
    features['min'] = data.min()
    features['max'] = data.max()
    features['range'] = features['max'] - features['min']
    
    # Percentiles
    features['q25'] = data.quantile(0.25)
    features['q75'] = data.quantile(0.75)
    features['iqr'] = features['q75'] - features['q25']
    
    # Coefficient of variation
    if features['mean'] != 0:
        features['cv'] = features['std'] / features['mean']
    else:
        features['cv'] = 0
    
    # Skewness and kurtosis
    features['skewness'] = data.skew()
    features['kurtosis'] = data.kurtosis()
    
    # Rolling statistics
    if len(data) >= window:
        rolling_data = data.rolling(window=window, min_periods=1)
        features[f'rolling_mean_{window}'] = rolling_data.mean().iloc[-1]
        features[f'rolling_std_{window}'] = rolling_data.std().iloc[-1]
        features[f'rolling_min_{window}'] = rolling_data.min().iloc[-1]
        features[f'rolling_max_{window}'] = rolling_data.max().iloc[-1]
    
    return features


def detect_outliers(data: pd.Series, method: str = "iqr", threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in data series.
    
    Args:
        data: Data series to analyze
        method: Outlier detection method ("iqr" or "zscore")
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    if method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == "zscore":
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")


def clean_data_series(data: pd.Series, remove_outliers: bool = False, 
                     fill_method: str = "forward") -> pd.Series:
    """
    Clean data series by handling missing values and outliers.
    
    Args:
        data: Data series to clean
        remove_outliers: Whether to remove outliers
        fill_method: Method for filling missing values ("forward", "backward", "mean", "median")
        
    Returns:
        Cleaned data series
    """
    cleaned_data = data.copy()
    
    # Handle missing values
    if fill_method == "forward":
        cleaned_data = cleaned_data.ffill()
    elif fill_method == "backward":
        cleaned_data = cleaned_data.bfill()
    elif fill_method == "mean":
        cleaned_data = cleaned_data.fillna(cleaned_data.mean())
    elif fill_method == "median":
        cleaned_data = cleaned_data.fillna(cleaned_data.median())
    
    # Remove outliers if requested
    if remove_outliers:
        outliers = detect_outliers(cleaned_data)
        cleaned_data = cleaned_data[~outliers]
    
    return cleaned_data


# =============================================================================
# DATA QUALITY AND CLEANING FUNCTIONS
# =============================================================================

def validate_data_quality(data: pd.DataFrame, required_columns: List[str], 
                         min_rows: int = 1) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required columns
        min_rows: Minimum number of rows required
        
    Returns:
        Dictionary with data quality metrics
        
    Raises:
        DataValidationError: If data quality is insufficient
    """
    quality_report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_columns': [],
        'missing_values': {},
        'duplicate_rows': 0,
        'data_types': {},
        'quality_score': 0.0
    }
    
    # Check required columns
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        quality_report['missing_columns'] = list(missing_columns)
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    
    # Check minimum rows
    if len(data) < min_rows:
        raise DataValidationError(f"Insufficient data: {len(data)} rows, minimum {min_rows} required")
    
    # Check for missing values
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            quality_report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_count / len(data) * 100)
            }
    
    # Check for duplicate rows
    quality_report['duplicate_rows'] = int(data.duplicated().sum())
    
    # Data types
    quality_report['data_types'] = {col: str(dtype) for col, dtype in data.dtypes.items()}
    
    # Calculate quality score (0-100)
    total_cells = len(data) * len(data.columns)
    missing_cells = sum(data.isnull().sum())
    duplicate_penalty = quality_report['duplicate_rows'] / len(data) * 10
    
    quality_score = max(0, 100 - (missing_cells / total_cells * 100) - duplicate_penalty)
    quality_report['quality_score'] = round(quality_score, 2)
    
    return quality_report


# =============================================================================
# FILE I/O OPERATIONS
# =============================================================================

def save_csv_file(data: List[Dict[str, Any]], filepath: str, columns: List[str]) -> None:
    """
    Save data to CSV file.
    
    Args:
        data: List of dictionaries to save
        filepath: Path to save the CSV file
        columns: List of column names in desired order
        
    Raises:
        CSVProcessingError: If file saving fails
    """
    try:
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            for row in data:
                # Ensure all required columns are present
                filtered_row = {col: row.get(col, '') for col in columns}
                writer.writerow(filtered_row)
                
        logger.info(f"Successfully saved {len(data)} rows to {filepath}")
        
    except Exception as e:
        raise CSVProcessingError(f"Error saving CSV file {filepath}: {str(e)}")


def load_csv_file(filepath: str, required_columns: List[str]) -> List[Dict[str, Any]]:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        required_columns: List of required column names
        
    Returns:
        List of dictionaries representing CSV rows
        
    Raises:
        CSVProcessingError: If file loading fails
    """
    try:
        if not Path(filepath).exists():
            raise CSVProcessingError(f"CSV file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            content = csvfile.read()
            
        return read_csv_to_dict(content, required_columns)
        
    except Exception as e:
        raise CSVProcessingError(f"Error loading CSV file {filepath}: {str(e)}")