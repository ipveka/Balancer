"""
Demand Domain Service Layer

This module provides the main service interface for demand forecasting,
including time series analysis and ML-based predictions.
All functions work with pandas DataFrames as the primary data format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
import logging

from .models import (
    DemandDataInput, ForecastOutput, ForecastAccuracy,
    SeasonalityPattern, TrendDirection, DemandAnalytics
)
from utils.ml_utils import (
    prepare_time_series_features, evaluate_forecast_accuracy,
    detect_demand_pattern
)

logger = logging.getLogger(__name__)


class DemandService:
    """
    Main service class for demand forecasting operations.
    
    This service handles demand forecasting, pattern analysis, and trend detection
    using pandas DataFrames as the primary data format.
    """
    
    def __init__(self):
        """Initialize the demand service."""
        self.logger = logging.getLogger(__name__)
    
    # =============================================================================
    # CORE DEMAND FORECASTING
    # =============================================================================
    
    def generate_forecast(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate demand forecasts using machine learning algorithms.
        
        Args:
            data (pd.DataFrame): Demand data with required columns:
                - date: Date in YYYY-MM-DD format
                - sku: Stock Keeping Unit identifier
                - quantity: Demand quantity
            params (Dict[str, Any], optional): Forecasting parameters
                
        Returns:
            Dict[str, Any]: Forecast results with predictions DataFrame
        """
        # Validate input data
        required_columns = ['date', 'sku', 'quantity']
        self._validate_dataframe_columns(data, required_columns)
        
        # Set default parameters
        if params is None:
            params = {}
        
        forecast_params = {
            'forecast_horizon_weeks': params.get('forecast_horizon_weeks', 12),
            'confidence_level': params.get('confidence_level', 0.95),
            'include_seasonality': params.get('include_seasonality', True),
            'algorithm': params.get('algorithm', 'lightgbm'),
            'auto_tune': params.get('auto_tune', True)
        }
        
        # Prepare data
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['sku', 'date']).reset_index(drop=True)
        
        # Validate minimum data requirements
        if len(df) < 28:
            raise ValueError("Minimum 28 data points required for reliable forecasting")
        
        # Process each SKU
        all_forecasts = []
        model_performance = {}
        processing_start = datetime.now()
        
        for sku in df['sku'].unique():
            try:
                sku_data = df[df['sku'] == sku].copy()
                
                # Generate features for ML model
                features_df = prepare_time_series_features(
                    sku_data, 'date', 'quantity',
                    lags=[1, 2, 3, 7, 14],
                    ma_windows=[3, 7, 14]
                )
                
                # Generate forecasts
                forecasts = self._generate_sku_forecast(
                    sku, sku_data, features_df, forecast_params
                )
                
                all_forecasts.extend(forecasts)
                
                # Calculate model performance for this SKU
                if len(sku_data) >= 56:  # Need enough data for validation
                    performance = self._calculate_model_performance(sku_data)
                    model_performance[sku] = performance
                
            except Exception as e:
                self.logger.error(f"Error forecasting SKU {sku}: {str(e)}")
                continue
        
        # Create results DataFrame
        forecasts_df = pd.DataFrame(all_forecasts)
        
        # Calculate overall performance metrics
        processing_time = (datetime.now() - processing_start).total_seconds()
        avg_performance = self._calculate_average_performance(model_performance)
        
        # Generate forecast summary
        forecast_summary = self._generate_forecast_summary(forecasts_df, df)
        
        return {
            'success': True,
            'forecasts': forecasts_df,
            'forecasts_count': len(all_forecasts),
            'processing_time_seconds': round(processing_time, 3),
            'model_performance': avg_performance,
            'forecast_summary': forecast_summary,
            'parameters_used': forecast_params
        }
    
    def analyze_demand_patterns(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze historical demand patterns and trends.
        
        Args:
            data (pd.DataFrame): Demand data with columns:
                - date: Date in YYYY-MM-DD format
                - sku: Stock Keeping Unit identifier
                - quantity: Demand quantity
            params (Dict[str, Any], optional): Analysis parameters
                
        Returns:
            Dict[str, Any]: Pattern analysis results
        """
        required_columns = ['date', 'sku', 'quantity']
        self._validate_dataframe_columns(data, required_columns)
        
        if params is None:
            params = {}
        
        analysis_params = {
            'include_seasonality': params.get('include_seasonality', True),
            'detect_outliers': params.get('detect_outliers', True),
            'trend_analysis': params.get('trend_analysis', True)
        }
        
        # Prepare data
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['sku', 'date']).reset_index(drop=True)
        
        # Analyze each SKU
        sku_analyses = []
        
        for sku in df['sku'].unique():
            try:
                sku_data = df[df['sku'] == sku].copy()
                
                if len(sku_data) < 4:
                    continue  # Need minimum data for analysis
                
                # Detect demand patterns using ML utils
                pattern_analysis = detect_demand_pattern(sku_data['quantity'].tolist())
                
                # Calculate additional metrics
                analysis_period_start = sku_data['date'].min().date()
                analysis_period_end = sku_data['date'].max().date()
                total_demand = sku_data['quantity'].sum()
                
                # Calculate time-based metrics
                period_days = (analysis_period_end - analysis_period_start).days + 1
                average_daily_demand = total_demand / period_days if period_days > 0 else 0
                
                # Statistical metrics
                demand_variance = sku_data['quantity'].var()
                demand_std_dev = sku_data['quantity'].std()
                coefficient_of_variation = demand_std_dev / sku_data['quantity'].mean() if sku_data['quantity'].mean() > 0 else 0
                
                # Find peak and low demand
                peak_idx = sku_data['quantity'].idxmax()
                low_idx = sku_data['quantity'].idxmin()
                
                # Detect outliers
                outlier_count = 0
                if analysis_params['detect_outliers']:
                    from utils.helpers import detect_outliers
                    outliers = detect_outliers(sku_data['quantity'], method='iqr')
                    outlier_count = outliers.sum()
                
                # Calculate data quality score
                data_quality_score = self._calculate_data_quality_score(sku_data)
                
                # Map pattern analysis to enum values
                trend_direction = self._map_trend_direction(pattern_analysis['trend'])
                seasonality_pattern = self._map_seasonality_pattern(pattern_analysis['seasonality'])
                
                analysis = {
                    'sku': sku,
                    'analysis_period_start': analysis_period_start,
                    'analysis_period_end': analysis_period_end,
                    'total_demand': int(total_demand),
                    'average_daily_demand': round(average_daily_demand, 2),
                    'demand_variance': round(demand_variance, 2),
                    'demand_std_dev': round(demand_std_dev, 2),
                    'coefficient_of_variation': round(coefficient_of_variation, 3),
                    'trend_direction': trend_direction,
                    'trend_coefficient': round(pattern_analysis['trend_coefficient'], 4),
                    'seasonality_pattern': seasonality_pattern,
                    'peak_demand_date': sku_data.loc[peak_idx, 'date'].date(),
                    'peak_demand_value': int(sku_data.loc[peak_idx, 'quantity']),
                    'low_demand_date': sku_data.loc[low_idx, 'date'].date(),
                    'low_demand_value': int(sku_data.loc[low_idx, 'quantity']),
                    'outlier_count': int(outlier_count),
                    'data_quality_score': round(data_quality_score, 3),
                    'volatility_level': pattern_analysis['volatility'],
                    'pattern_strength': round(pattern_analysis['pattern_strength'], 3)
                }
                
                sku_analyses.append(analysis)
                
            except Exception as e:
                self.logger.error(f"Error analyzing SKU {sku}: {str(e)}")
                continue
        
        # Create results DataFrame
        analyses_df = pd.DataFrame(sku_analyses)
        
        # Generate summary statistics
        summary_stats = self._generate_analysis_summary(analyses_df)
        
        return {
            'success': True,
            'analyses': analyses_df,
            'analyses_count': len(sku_analyses),
            'summary_statistics': summary_stats,
            'parameters_used': analysis_params
        }
    
    def validate_forecast_accuracy(
        self, 
        historical_data: pd.DataFrame, 
        forecast_data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate forecast accuracy using historical data.
        
        Args:
            historical_data (pd.DataFrame): Historical actual demand data
            forecast_data (pd.DataFrame): Historical forecast data
            params (Dict[str, Any], optional): Validation parameters
                
        Returns:
            Dict[str, Any]: Accuracy validation results
        """
        required_columns_historical = ['date', 'sku', 'quantity']
        required_columns_forecast = ['date', 'sku', 'prediction']
        
        self._validate_dataframe_columns(historical_data, required_columns_historical)
        self._validate_dataframe_columns(forecast_data, required_columns_forecast)
        
        if params is None:
            params = {}
        
        validation_params = {
            'metrics': params.get('metrics', ['mae', 'mape', 'rmse']),
            'confidence_intervals': params.get('confidence_intervals', True)
        }
        
        # Prepare data
        historical_df = historical_data.copy()
        forecast_df = forecast_data.copy()
        
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        # Merge actual and forecast data
        merged_df = pd.merge(
            historical_df, forecast_df,
            on=['date', 'sku'],
            suffixes=('_actual', '_forecast')
        )
        
        if merged_df.empty:
            raise ValueError("No matching dates/SKUs found between historical and forecast data")
        
        # Calculate accuracy metrics for each SKU
        sku_accuracies = []
        
        for sku in merged_df['sku'].unique():
            sku_data = merged_df[merged_df['sku'] == sku]
            
            if len(sku_data) < 2:
                continue
            
            # Calculate accuracy metrics using ML utils
            accuracy_metrics = evaluate_forecast_accuracy(
                sku_data['quantity'].values,
                sku_data['prediction'].values
            )
            
            # Calculate additional metrics
            forecast_bias = (sku_data['prediction'] - sku_data['quantity']).mean()
            accuracy_percentage = max(0, 100 - accuracy_metrics['mape'])
            
            sku_accuracy = {
                'sku': sku,
                'evaluation_period_start': sku_data['date'].min().date(),
                'evaluation_period_end': sku_data['date'].max().date(),
                'total_forecasts_evaluated': len(sku_data),
                'mean_absolute_error': round(accuracy_metrics['mae'], 2),
                'mean_absolute_percentage_error': round(accuracy_metrics['mape'], 2),
                'root_mean_square_error': round(accuracy_metrics['rmse'], 2),
                'mean_forecast_error': round(forecast_bias, 2),
                'forecast_accuracy_percentage': round(accuracy_percentage, 2),
                'directional_accuracy': round(accuracy_metrics['directional_accuracy'], 2)
            }
            
            sku_accuracies.append(sku_accuracy)
        
        # Create results DataFrame
        accuracies_df = pd.DataFrame(sku_accuracies)
        
        # Calculate overall accuracy metrics
        overall_metrics = self._calculate_overall_accuracy(accuracies_df)
        
        return {
            'success': True,
            'accuracy_results': accuracies_df,
            'overall_metrics': overall_metrics,
            'validation_summary': {
                'skus_evaluated': len(sku_accuracies),
                'total_forecasts': accuracies_df['total_forecasts_evaluated'].sum(),
                'avg_accuracy_percentage': accuracies_df['forecast_accuracy_percentage'].mean(),
                'best_performing_sku': accuracies_df.loc[
                    accuracies_df['forecast_accuracy_percentage'].idxmax(), 'sku'
                ] if not accuracies_df.empty else None
            }
        }
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    def _validate_dataframe_columns(self, df: pd.DataFrame, required_columns: List[str]):
        """Validate that DataFrame contains required columns."""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
    
    def _generate_sku_forecast(
        self, 
        sku: str, 
        sku_data: pd.DataFrame, 
        features_df: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate forecast for a single SKU."""
        forecasts = []
        
        # Simple trend-based forecasting (in production, use LightGBM or similar)
        last_date = sku_data['date'].max()
        recent_avg = sku_data['quantity'].tail(7).mean()
        trend = sku_data['quantity'].diff().tail(7).mean()
        
        # Generate forecasts for specified horizon
        for week in range(1, params['forecast_horizon_weeks'] + 1):
            forecast_date = last_date + timedelta(weeks=week)
            
            # Simple trend projection with seasonality
            base_prediction = recent_avg + (trend * week * 7)
            
            # Add seasonal component (simplified)
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * week / 52)
            prediction = max(0, base_prediction * seasonal_factor)
            
            # Calculate confidence intervals
            std_dev = sku_data['quantity'].std()
            confidence_width = 1.96 * std_dev  # 95% confidence interval
            
            confidence_interval_lower = max(0, prediction - confidence_width)
            confidence_interval_upper = prediction + confidence_width
            
            # Determine forecast accuracy level
            cv = std_dev / sku_data['quantity'].mean() if sku_data['quantity'].mean() > 0 else 1
            if cv < 0.1:
                accuracy = ForecastAccuracy.HIGH
            elif cv < 0.2:
                accuracy = ForecastAccuracy.MEDIUM
            else:
                accuracy = ForecastAccuracy.LOW
            
            forecast = {
                'sku': sku,
                'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                'prediction': round(prediction, 2),
                'confidence_interval_lower': round(confidence_interval_lower, 2),
                'confidence_interval_upper': round(confidence_interval_upper, 2),
                'confidence_level': params['confidence_level'],
                'forecast_accuracy': accuracy.value,
                'model_version': 'v2.1.0'
            }
            
            forecasts.append(forecast)
        
        return forecasts
    
    def _calculate_model_performance(self, sku_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate model performance metrics for backtesting."""
        # Simple backtesting using last 25% of data
        split_point = int(len(sku_data) * 0.75)
        train_data = sku_data.iloc[:split_point]
        test_data = sku_data.iloc[split_point:]
        
        if len(test_data) < 2:
            return {'mae': 0, 'mape': 0, 'rmse': 0, 'directional_accuracy': 0}
        
        # Simple prediction using trend from training data
        trend = train_data['quantity'].diff().mean()
        last_value = train_data['quantity'].iloc[-1]
        
        predictions = []
        for i in range(len(test_data)):
            pred = max(0, last_value + trend * (i + 1))
            predictions.append(pred)
        
        # Calculate accuracy metrics
        accuracy_metrics = evaluate_forecast_accuracy(
            test_data['quantity'].values,
            np.array(predictions)
        )
        
        return accuracy_metrics
    
    def _calculate_average_performance(self, model_performance: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate average performance across all SKUs."""
        if not model_performance:
            return {'mae': 0, 'mape': 0, 'rmse': 0, 'directional_accuracy': 0}
        
        metrics = ['mae', 'mape', 'rmse', 'directional_accuracy']
        avg_performance = {}
        
        for metric in metrics:
            values = [perf[metric] for perf in model_performance.values() if metric in perf]
            avg_performance[metric] = np.mean(values) if values else 0
        
        return avg_performance
    
    def _generate_forecast_summary(self, forecasts_df: pd.DataFrame, historical_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of forecast results."""
        if forecasts_df.empty:
            return {}
        
        total_forecasted_demand = forecasts_df['prediction'].sum()
        
        # Detect overall trend
        recent_demand = historical_df.groupby('date')['quantity'].sum().tail(14).mean()
        forecast_demand = forecasts_df.groupby('forecast_date')['prediction'].sum().head(14).mean()
        
        if forecast_demand > recent_demand * 1.05:
            trend_direction = 'INCREASING'
        elif forecast_demand < recent_demand * 0.95:
            trend_direction = 'DECREASING'
        else:
            trend_direction = 'STABLE'
        
        # Check for seasonality detection
        seasonality_detected = len(forecasts_df['sku'].unique()) > 1  # Simplified check
        
        # Calculate confidence score
        avg_confidence = forecasts_df['confidence_level'].mean()
        high_accuracy_count = len(forecasts_df[forecasts_df['forecast_accuracy'] == 'HIGH'])
        confidence_score = (high_accuracy_count / len(forecasts_df)) * avg_confidence
        
        return {
            'total_forecasted_demand': round(total_forecasted_demand, 2),
            'trend_direction': trend_direction,
            'seasonality_detected': seasonality_detected,
            'confidence_score': round(confidence_score, 3),
            'forecast_horizon_weeks': len(forecasts_df['forecast_date'].unique()),
            'skus_forecasted': len(forecasts_df['sku'].unique())
        }
    
    def _map_trend_direction(self, trend_str: str) -> str:
        """Map trend analysis result to enum value."""
        if trend_str == 'increasing':
            return TrendDirection.INCREASING.value
        elif trend_str == 'decreasing':
            return TrendDirection.DECREASING.value
        elif trend_str == 'stable':
            return TrendDirection.STABLE.value
        else:
            return TrendDirection.VOLATILE.value
    
    def _map_seasonality_pattern(self, seasonality_str: str) -> str:
        """Map seasonality analysis result to enum value."""
        if seasonality_str == 'seasonal':
            return SeasonalityPattern.YEARLY.value
        else:
            return SeasonalityPattern.NONE.value
    
    def _calculate_data_quality_score(self, sku_data: pd.DataFrame) -> float:
        """Calculate data quality score for SKU data."""
        # Check for missing values
        missing_ratio = sku_data['quantity'].isnull().sum() / len(sku_data)
        
        # Check for zero values
        zero_ratio = (sku_data['quantity'] == 0).sum() / len(sku_data)
        
        # Check for negative values
        negative_ratio = (sku_data['quantity'] < 0).sum() / len(sku_data)
        
        # Calculate quality score
        quality_score = 1.0 - (missing_ratio + zero_ratio * 0.5 + negative_ratio)
        
        return max(0.0, min(1.0, quality_score))
    
    def _generate_analysis_summary(self, analyses_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from pattern analyses."""
        if analyses_df.empty:
            return {}
        
        return {
            'total_skus_analyzed': len(analyses_df),
            'avg_coefficient_variation': analyses_df['coefficient_of_variation'].mean(),
            'seasonal_items_count': len(analyses_df[analyses_df['seasonality_pattern'] != 'NONE']),
            'high_variability_items': len(analyses_df[analyses_df['coefficient_of_variation'] > 0.5]),
            'trending_up_items': len(analyses_df[analyses_df['trend_direction'] == 'INCREASING']),
            'trending_down_items': len(analyses_df[analyses_df['trend_direction'] == 'DECREASING']),
            'avg_data_quality_score': analyses_df['data_quality_score'].mean(),
            'total_demand_analyzed': analyses_df['total_demand'].sum()
        }
    
    def _calculate_overall_accuracy(self, accuracies_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall accuracy metrics."""
        if accuracies_df.empty:
            return {}
        
        return {
            'overall_mae': accuracies_df['mean_absolute_error'].mean(),
            'overall_mape': accuracies_df['mean_absolute_percentage_error'].mean(),
            'overall_rmse': accuracies_df['root_mean_square_error'].mean(),
            'overall_accuracy_percentage': accuracies_df['forecast_accuracy_percentage'].mean(),
            'overall_directional_accuracy': accuracies_df['directional_accuracy'].mean()
        }
    
    # =============================================================================
    # DATA FORMAT CONVERSION UTILITIES
    # =============================================================================
    
    def from_csv(self, csv_data: str) -> pd.DataFrame:
        """Convert CSV string to DataFrame for processing."""
        from io import StringIO
        
        try:
            df = pd.read_csv(StringIO(csv_data))
            return df
        except Exception as e:
            raise ValueError(f"Invalid CSV format: {str(e)}")
    
    def to_csv(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to CSV string."""
        return df.to_csv(index=False)
    
    def from_json(self, json_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert JSON data to DataFrame for processing."""
        return pd.DataFrame(json_data)
    
    def to_json(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to JSON format."""
        return df.to_dict('records')