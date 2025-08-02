"""
Machine learning utilities for distance calculation, optimization algorithms, 
and model training utilities.

This module provides ML-specific functions used across domain modules for
optimization, forecasting, and route planning algorithms.

Key Features:
- Geographic distance calculations (Haversine, Euclidean)
- Vehicle Routing Problem (VRP) optimization algorithms
- Inventory and production optimization
- Time series feature engineering and validation
- Statistical analysis and pattern detection
- Cross-validation for time series data

Algorithms Included:
- VRP: Greedy assignment, Nearest neighbor
- Optimization: EOQ, EBQ, Safety stock calculation
- ML: Feature engineering, accuracy evaluation
- Statistics: Pattern detection, outlier analysis

Author: Balancer Platform
Version: 1.0.0
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                           mean_squared_error)
from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class MLUtilsError(Exception):
    """
    Custom exception for ML utilities errors.
    
    Raised when machine learning operations, optimizations, or 
    calculations fail due to invalid inputs or computational issues.
    """
    pass


# =============================================================================
# GEOGRAPHIC DISTANCE CALCULATION FUNCTIONS
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.
    
    This function calculates the shortest distance between two points on the surface
    of a sphere (Earth), taking into account the Earth's curvature. More accurate
    than Euclidean distance for geographic coordinates.
    
    Args:
        lat1 (float): Latitude of first point in decimal degrees
        lon1 (float): Longitude of first point in decimal degrees
        lat2 (float): Latitude of second point in decimal degrees
        lon2 (float): Longitude of second point in decimal degrees
        
    Returns:
        float: Distance in kilometers
        
    Example:
        >>> # Distance between New York and Los Angeles
        >>> dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        >>> 3900 < dist < 4000  # Approximately 3935 km
        True
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    
    return c * r


def euclidean_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Euclidean distance between two points (approximation for short distances).
    
    Args:
        lat1, lon1: Latitude and longitude of first point in decimal degrees
        lat2, lon2: Latitude and longitude of second point in decimal degrees
        
    Returns:
        Distance in kilometers (approximate)
    """
    # Convert to approximate km (rough approximation)
    lat_diff = (lat2 - lat1) * 111.0  # 1 degree lat ≈ 111 km
    lon_diff = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    
    return math.sqrt(lat_diff**2 + lon_diff**2)


def create_distance_matrix(locations: List[Dict[str, float]], 
                          method: str = "haversine") -> np.ndarray:
    """
    Create distance matrix for a list of locations.
    
    Args:
        locations: List of dictionaries with 'lat' and 'lon' keys
        method: Distance calculation method ("haversine" or "euclidean")
        
    Returns:
        Square distance matrix as numpy array
        
    Raises:
        MLUtilsError: If invalid method or locations provided
    """
    if not locations:
        raise MLUtilsError("No locations provided")
    
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    
    distance_func = haversine_distance if method == "haversine" else euclidean_distance
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_func(
                locations[i]['lat'], locations[i]['lon'],
                locations[j]['lat'], locations[j]['lon']
            )
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist  # Symmetric matrix
    
    return distance_matrix


# =============================================================================
# VEHICLE ROUTING PROBLEM (VRP) ALGORITHMS
# =============================================================================

class VRPSolver:
    """
    Vehicle Routing Problem solver with multiple heuristic algorithms.
    
    This class implements various algorithms to solve the Vehicle Routing Problem,
    which involves finding optimal routes for a fleet of vehicles to serve
    customers while respecting capacity constraints.
    
    Supported Algorithms:
    - Greedy assignment: Assigns closest feasible customers to vehicles
    - Nearest neighbor: Builds routes by always choosing nearest unvisited customer
    
    Attributes:
        distance_matrix (np.ndarray): Square matrix of distances between all locations
        demands (List[float]): Customer demands (excluding depot)
        vehicle_capacities (List[float]): Maximum capacity for each vehicle
        depot_index (int): Index of depot location in distance matrix
        num_locations (int): Total number of locations including depot
        num_vehicles (int): Number of available vehicles
    """
    
    def __init__(self, distance_matrix: np.ndarray, demands: List[float], 
                 vehicle_capacities: List[float], depot_index: int = 0):
        """
        Initialize VRP solver with problem parameters.
        
        Args:
            distance_matrix (np.ndarray): Square matrix of distances between locations
            demands (List[float]): List of demands for each customer location
            vehicle_capacities (List[float]): Maximum capacity for each vehicle
            depot_index (int, optional): Index of depot location. Defaults to 0.
            
        Raises:
            MLUtilsError: If input parameters are invalid or inconsistent
        """
        # Validate inputs
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise MLUtilsError("Distance matrix must be square")
        if len(demands) >= len(distance_matrix):
            raise MLUtilsError("Number of demands must be less than number of locations")
        if not vehicle_capacities:
            raise MLUtilsError("At least one vehicle must be provided")
        if depot_index >= len(distance_matrix):
            raise MLUtilsError("Depot index out of bounds")
            
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacities = vehicle_capacities
        self.depot_index = depot_index
        self.num_locations = len(distance_matrix)
        self.num_vehicles = len(vehicle_capacities)
    
    def greedy_assignment(self) -> Dict[str, Any]:
        """
        Solve VRP using greedy nearest neighbor algorithm.
        
        Returns:
            Dictionary with solution details including routes and total distance
        """
        routes = []
        total_distance = 0.0
        unassigned_locations = list(range(self.num_locations))
        unassigned_locations.remove(self.depot_index)  # Remove depot
        
        for vehicle_idx in range(self.num_vehicles):
            if not unassigned_locations:
                break
                
            route = [self.depot_index]
            current_location = self.depot_index
            current_capacity = self.vehicle_capacities[vehicle_idx]
            route_distance = 0.0
            
            while unassigned_locations:
                # Find nearest feasible location
                best_location = None
                best_distance = float('inf')
                
                for location in unassigned_locations:
                    # Check capacity constraint
                    demand_idx = location - 1 if location > self.depot_index else location
                    if demand_idx < len(self.demands) and self.demands[demand_idx] <= current_capacity:
                        distance = self.distance_matrix[current_location][location]
                        if distance < best_distance:
                            best_distance = distance
                            best_location = location
                
                if best_location is None:
                    break  # No feasible location found
                
                # Add location to route
                route.append(best_location)
                route_distance += best_distance
                current_location = best_location
                unassigned_locations.remove(best_location)
                
                # Update capacity
                demand_idx = best_location - 1 if best_location > self.depot_index else best_location
                if demand_idx < len(self.demands):
                    current_capacity -= self.demands[demand_idx]
            
            # Return to depot
            if len(route) > 1:
                route.append(self.depot_index)
                route_distance += self.distance_matrix[current_location][self.depot_index]
                routes.append({
                    'vehicle_id': vehicle_idx,
                    'route': route,
                    'distance': route_distance
                })
                total_distance += route_distance
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'unassigned_locations': unassigned_locations,
            'algorithm': 'greedy'
        }
    
    def nearest_neighbor_assignment(self) -> Dict[str, Any]:
        """
        Solve VRP using nearest neighbor heuristic with capacity constraints.
        
        Returns:
            Dictionary with solution details including routes and total distance
        """
        routes = []
        total_distance = 0.0
        unassigned_locations = list(range(self.num_locations))
        unassigned_locations.remove(self.depot_index)
        
        for vehicle_idx in range(self.num_vehicles):
            if not unassigned_locations:
                break
                
            route = [self.depot_index]
            current_location = self.depot_index
            current_capacity = self.vehicle_capacities[vehicle_idx]
            route_distance = 0.0
            
            # Start with nearest location to depot
            if unassigned_locations:
                nearest_to_depot = min(unassigned_locations, 
                                     key=lambda x: self.distance_matrix[self.depot_index][x])
                
                demand_idx = nearest_to_depot - 1 if nearest_to_depot > self.depot_index else nearest_to_depot
                if demand_idx < len(self.demands) and self.demands[demand_idx] <= current_capacity:
                    route.append(nearest_to_depot)
                    route_distance += self.distance_matrix[self.depot_index][nearest_to_depot]
                    current_location = nearest_to_depot
                    unassigned_locations.remove(nearest_to_depot)
                    current_capacity -= self.demands[demand_idx]
            
            # Continue with nearest neighbor
            while unassigned_locations:
                best_location = None
                best_distance = float('inf')
                
                for location in unassigned_locations:
                    demand_idx = location - 1 if location > self.depot_index else location
                    if demand_idx < len(self.demands) and self.demands[demand_idx] <= current_capacity:
                        distance = self.distance_matrix[current_location][location]
                        if distance < best_distance:
                            best_distance = distance
                            best_location = location
                
                if best_location is None:
                    break
                
                route.append(best_location)
                route_distance += best_distance
                current_location = best_location
                unassigned_locations.remove(best_location)
                
                demand_idx = best_location - 1 if best_location > self.depot_index else best_location
                if demand_idx < len(self.demands):
                    current_capacity -= self.demands[demand_idx]
            
            # Return to depot
            if len(route) > 1:
                route.append(self.depot_index)
                route_distance += self.distance_matrix[current_location][self.depot_index]
                routes.append({
                    'vehicle_id': vehicle_idx,
                    'route': route,
                    'distance': route_distance
                })
                total_distance += route_distance
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'unassigned_locations': unassigned_locations,
            'algorithm': 'nearest_neighbor'
        }


# =============================================================================
# INVENTORY AND PRODUCTION OPTIMIZATION ALGORITHMS
# =============================================================================

def optimize_inventory_levels(current_stock: int, demand_mean: float, demand_std: float,
                            lead_time: int, service_level: float = 0.95) -> Dict[str, float]:
    """
    Optimize inventory levels using statistical methods.
    
    Args:
        current_stock: Current inventory level
        demand_mean: Average demand per period
        demand_std: Standard deviation of demand
        lead_time: Lead time in periods
        service_level: Target service level (0-1)
        
    Returns:
        Dictionary with optimized inventory parameters
    """
    from scipy.stats import norm
    
    # Calculate safety stock
    z_score = norm.ppf(service_level)
    lead_time_demand_std = demand_std * math.sqrt(lead_time)
    safety_stock = z_score * lead_time_demand_std
    
    # Calculate reorder point
    lead_time_demand = demand_mean * lead_time
    reorder_point = lead_time_demand + safety_stock
    
    # Calculate optimal order quantity (EOQ approximation)
    # Simplified EOQ assuming holding cost = 20% and ordering cost = $100
    holding_cost_rate = 0.20
    ordering_cost = 100.0
    annual_demand = demand_mean * 52  # Assuming weekly demand
    
    if annual_demand > 0 and holding_cost_rate > 0:
        eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_rate)
    else:
        eoq = demand_mean * 4  # 4 weeks of demand as fallback
    
    return {
        'safety_stock': max(0, safety_stock),
        'reorder_point': max(0, reorder_point),
        'optimal_order_quantity': max(1, eoq),
        'current_stock': current_stock,
        'service_level_achieved': service_level,
        'z_score': z_score
    }


def optimize_production_batch(demand_forecast: List[float], setup_cost: float,
                            holding_cost_rate: float, production_rate: float) -> Dict[str, Any]:
    """
    Optimize production batch sizes using economic batch quantity model.
    
    Args:
        demand_forecast: List of forecasted demand values
        setup_cost: Fixed cost per production setup
        holding_cost_rate: Holding cost rate per unit per period
        production_rate: Production rate (units per period)
        
    Returns:
        Dictionary with optimized batch parameters
    """
    total_demand = sum(demand_forecast)
    avg_demand = total_demand / len(demand_forecast) if demand_forecast else 0
    
    if avg_demand <= 0:
        return {
            'optimal_batch_size': 0,
            'number_of_batches': 0,
            'total_cost': 0,
            'setup_cost_total': 0,
            'holding_cost_total': 0
        }
    
    # Economic Batch Quantity (EBQ) formula
    if holding_cost_rate > 0 and production_rate > avg_demand:
        ebq = math.sqrt((2 * total_demand * setup_cost) / 
                       (holding_cost_rate * (1 - avg_demand / production_rate)))
    else:
        ebq = math.sqrt((2 * total_demand * setup_cost) / holding_cost_rate) if holding_cost_rate > 0 else avg_demand
    
    # Calculate number of batches and costs
    optimal_batch_size = max(1, int(ebq))
    number_of_batches = math.ceil(total_demand / optimal_batch_size)
    
    setup_cost_total = number_of_batches * setup_cost
    avg_inventory = optimal_batch_size / 2
    holding_cost_total = avg_inventory * holding_cost_rate * len(demand_forecast)
    total_cost = setup_cost_total + holding_cost_total
    
    return {
        'optimal_batch_size': optimal_batch_size,
        'number_of_batches': number_of_batches,
        'total_cost': total_cost,
        'setup_cost_total': setup_cost_total,
        'holding_cost_total': holding_cost_total,
        'average_inventory': avg_inventory
    }


# =============================================================================
# MODEL TRAINING AND EVALUATION UTILITIES
# =============================================================================

def prepare_time_series_features(data: pd.DataFrame, date_col: str, value_col: str,
                               lags: List[int] = [1, 2, 3, 4], 
                               ma_windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
    """
    Prepare time series features for machine learning.
    
    Args:
        data: DataFrame with time series data
        date_col: Name of date column
        value_col: Name of value column
        lags: List of lag periods to create
        ma_windows: List of moving average windows
        
    Returns:
        DataFrame with engineered features
    """
    # Ensure data is sorted by date
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create lagged features
    for lag in lags:
        df[f'lag_{lag}'] = df[value_col].shift(lag)
    
    # Create moving averages
    for window in ma_windows:
        df[f'ma_{window}'] = df[value_col].rolling(window=window, min_periods=1).mean()
        df[f'ma_std_{window}'] = df[value_col].rolling(window=window, min_periods=1).std()
    
    # Create time-based features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['day_of_week'] = df[date_col].dt.dayofweek
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Trend feature
    df['trend'] = range(len(df))
    
    return df


def evaluate_forecast_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'mae': float('inf'),
            'rmse': float('inf'),
            'mape': float('inf'),
            'directional_accuracy': 0.0
        }
    
    # Ensure arrays are the same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE with handling for zero values
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Directional accuracy
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }


def time_series_cross_validation(data: pd.DataFrame, target_col: str, 
                               feature_cols: List[str], n_splits: int = 5) -> Dict[str, Any]:
    """
    Perform time series cross-validation.
    
    Args:
        data: DataFrame with time series data
        target_col: Name of target column
        feature_cols: List of feature column names
        n_splits: Number of cross-validation splits
        
    Returns:
        Dictionary with cross-validation results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    X = data[feature_cols].fillna(0)  # Fill NaN values
    y = data[target_col].fillna(0)
    
    cv_scores = {
        'mae_scores': [],
        'rmse_scores': [],
        'mape_scores': []
    }
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Simple linear regression for demonstration
        # In practice, this would use LightGBM or other ML models
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = evaluate_forecast_accuracy(y_test.values, y_pred)
        cv_scores['mae_scores'].append(metrics['mae'])
        cv_scores['rmse_scores'].append(metrics['rmse'])
        cv_scores['mape_scores'].append(metrics['mape'])
    
    # Calculate average scores
    cv_results = {
        'mean_mae': np.mean(cv_scores['mae_scores']),
        'std_mae': np.std(cv_scores['mae_scores']),
        'mean_rmse': np.mean(cv_scores['rmse_scores']),
        'std_rmse': np.std(cv_scores['rmse_scores']),
        'mean_mape': np.mean(cv_scores['mape_scores']),
        'std_mape': np.std(cv_scores['mape_scores']),
        'individual_scores': cv_scores
    }
    
    return cv_results


# =============================================================================
# STATISTICAL UTILITIES AND ANALYSIS FUNCTIONS
# =============================================================================

def calculate_safety_stock(demand_mean: float, demand_std: float, 
                         lead_time: int, service_level: float = 0.95) -> float:
    """
    Calculate safety stock using statistical method.
    
    Args:
        demand_mean: Average demand per period
        demand_std: Standard deviation of demand
        lead_time: Lead time in periods
        service_level: Target service level (0-1)
        
    Returns:
        Calculated safety stock level
    """
    from scipy.stats import norm
    
    z_score = norm.ppf(service_level)
    lead_time_demand_std = demand_std * math.sqrt(lead_time)
    safety_stock = z_score * lead_time_demand_std
    
    return max(0, safety_stock)


def calculate_reorder_point(demand_mean: float, lead_time: int, 
                          safety_stock: float) -> float:
    """
    Calculate reorder point.
    
    Args:
        demand_mean: Average demand per period
        lead_time: Lead time in periods
        safety_stock: Safety stock level
        
    Returns:
        Calculated reorder point
    """
    lead_time_demand = demand_mean * lead_time
    reorder_point = lead_time_demand + safety_stock
    
    return max(0, reorder_point)


def detect_demand_pattern(demand_data: List[float]) -> Dict[str, Any]:
    """
    Detect patterns in demand data (trend, seasonality, etc.).
    
    Args:
        demand_data: List of demand values
        
    Returns:
        Dictionary with pattern analysis results
    """
    if len(demand_data) < 4:
        return {
            'trend': 'insufficient_data',
            'seasonality': 'insufficient_data',
            'volatility': 'insufficient_data',
            'pattern_strength': 0.0
        }
    
    data = np.array(demand_data)
    
    # Trend analysis using linear regression
    x = np.arange(len(data))
    trend_coef = np.polyfit(x, data, 1)[0]
    
    if trend_coef > 0.1:
        trend = 'increasing'
    elif trend_coef < -0.1:
        trend = 'decreasing'
    else:
        trend = 'stable'
    
    # Seasonality detection (simplified)
    if len(data) >= 12:  # Need at least 12 points for monthly seasonality
        # Calculate autocorrelation at lag 12 (monthly)
        mean_val = np.mean(data)
        numerator = np.sum((data[:-12] - mean_val) * (data[12:] - mean_val))
        denominator = np.sum((data - mean_val) ** 2)
        
        if denominator > 0:
            autocorr_12 = numerator / denominator
            seasonality = 'seasonal' if abs(autocorr_12) > 0.3 else 'non_seasonal'
        else:
            seasonality = 'non_seasonal'
    else:
        seasonality = 'insufficient_data'
    
    # Volatility analysis
    cv = np.std(data) / np.mean(data) if np.mean(data) > 0 else float('inf')
    
    if cv < 0.2:
        volatility = 'low'
    elif cv < 0.5:
        volatility = 'medium'
    else:
        volatility = 'high'
    
    # Pattern strength (combination of trend and seasonality strength)
    pattern_strength = min(1.0, abs(trend_coef) + (abs(autocorr_12) if 'autocorr_12' in locals() else 0))
    
    return {
        'trend': trend,
        'trend_coefficient': trend_coef,
        'seasonality': seasonality,
        'volatility': volatility,
        'coefficient_of_variation': cv,
        'pattern_strength': pattern_strength
    }