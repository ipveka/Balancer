"""
Demand Domain Data Models

This module contains Pydantic models for the demand forecasting domain,
including demand records, forecast models, and analytics with comprehensive
validation rules and time period constraints.
"""

from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import re


class ForecastHorizon(str, Enum):
    """Enumeration for forecast horizon periods."""
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class ForecastAccuracy(str, Enum):
    """Enumeration for forecast accuracy levels."""
    HIGH = "HIGH"      # MAPE < 10%
    MEDIUM = "MEDIUM"  # MAPE 10-20%
    LOW = "LOW"        # MAPE > 20%


class SeasonalityPattern(str, Enum):
    """Enumeration for seasonality patterns."""
    NONE = "NONE"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    YEARLY = "YEARLY"


class TrendDirection(str, Enum):
    """Enumeration for trend directions."""
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"
    VOLATILE = "VOLATILE"


# Input Models for CSV Processing

class DemandDataInput(BaseModel):
    """
    Input model for demand data processing.
    
    Represents a single demand record with date, SKU, and quantity
    for time-series forecasting and analysis.
    """
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    sku: str = Field(..., description="Stock Keeping Unit identifier", min_length=1, max_length=50)
    quantity: int = Field(..., description="Demand quantity", ge=0)

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format is YYYY-MM-DD and is a valid date."""
        try:
            parsed_date = datetime.strptime(v, '%Y-%m-%d').date()
            # Check if date is not too far in the future (more than 1 year)
            if parsed_date > date.today() + timedelta(days=365):
                raise ValueError('Date cannot be more than 1 year in the future')
            # Check if date is not too far in the past (more than 10 years)
            if parsed_date < date.today() - timedelta(days=3650):
                raise ValueError('Date cannot be more than 10 years in the past')
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError('Date must be in YYYY-MM-DD format')
            raise e
        return v

    @field_validator('sku')
    @classmethod
    def validate_sku_format(cls, v):
        """Validate SKU format - alphanumeric with hyphens and underscores allowed."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('SKU must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()

    @field_validator('quantity')
    @classmethod
    def validate_quantity_range(cls, v):
        """Validate quantity is within reasonable range."""
        if v > 1000000:  # 1 million units max
            raise ValueError('Quantity cannot exceed 1,000,000 units')
        return v


class DemandDataCSV(BaseModel):
    """Container model for demand CSV data processing."""
    data: List[DemandDataInput] = Field(..., description="List of demand data records")

    @field_validator('data')
    @classmethod
    def validate_data_consistency(cls, v):
        """Validate data consistency and completeness."""
        if len(v) == 0:
            raise ValueError('Demand data cannot be empty')
        
        # Check for reasonable data volume (at least 4 weeks of data for forecasting)
        if len(v) < 28:
            raise ValueError('Minimum 28 data points (4 weeks) required for reliable forecasting')
        
        # Check for duplicate date-sku combinations
        date_sku_pairs = [(item.date, item.sku) for item in v]
        if len(date_sku_pairs) != len(set(date_sku_pairs)):
            raise ValueError('Duplicate date-SKU combinations found in demand data')
        
        return v


# Output Models

class ForecastOutput(BaseModel):
    """
    Output model for demand forecasts.
    
    Contains forecasted demand values with confidence intervals
    and accuracy metrics for decision making.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    forecast_date: str = Field(..., description="Forecast date in YYYY-MM-DD format")
    prediction: float = Field(..., description="Forecasted demand quantity", ge=0)
    confidence_interval_lower: float = Field(..., description="Lower bound of confidence interval", ge=0)
    confidence_interval_upper: float = Field(..., description="Upper bound of confidence interval", ge=0)
    confidence_level: float = Field(0.95, description="Confidence level (default 95%)", gt=0, le=1)
    forecast_accuracy: ForecastAccuracy = Field(..., description="Forecast accuracy level")
    model_version: str = Field(..., description="Model version used for prediction")

    @field_validator('forecast_date')
    @classmethod
    def validate_forecast_date_format(cls, v):
        """Validate forecast date format is YYYY-MM-DD."""
        try:
            forecast_dt = datetime.strptime(v, '%Y-%m-%d').date()
            # Forecast should be for future dates
            if forecast_dt <= date.today():
                raise ValueError('Forecast date must be in the future')
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError('Forecast date must be in YYYY-MM-DD format')
            raise e
        return v

    @model_validator(mode='after')
    def validate_confidence_interval(self):
        """Validate confidence interval bounds."""
        if self.confidence_interval_lower > self.confidence_interval_upper:
            raise ValueError('Lower confidence bound must be less than upper bound')
        
        if self.prediction < self.confidence_interval_lower or self.prediction > self.confidence_interval_upper:
            raise ValueError('Prediction must be within confidence interval bounds')
        
        return self


class ForecastOutputCSV(BaseModel):
    """Container model for forecast output CSV."""
    forecasts: List[ForecastOutput] = Field(..., description="List of demand forecasts")
    generated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when forecasts were generated")
    total_forecasts: int = Field(..., description="Total number of forecasts")
    model_performance: Dict[str, float] = Field(default_factory=dict, description="Overall model performance metrics")
    
    @model_validator(mode='after')
    def validate_count_matches(self):
        """Ensure total count matches actual forecasts."""
        if self.total_forecasts != len(self.forecasts):
            raise ValueError('Total forecasts count does not match actual forecasts')
        return self


# Internal Processing Models

class DemandRecord(BaseModel):
    """
    Internal model for demand record processing.
    
    Contains processed demand data with engineered features
    for machine learning model training and inference.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    date: datetime = Field(..., description="Demand date")
    quantity: int = Field(..., description="Actual demand quantity", ge=0)
    processed_features: Optional[Dict[str, float]] = Field(default_factory=dict, description="Engineered features for ML")
    is_outlier: bool = Field(False, description="Whether this record is identified as an outlier")
    seasonality_factor: Optional[float] = Field(None, description="Seasonality adjustment factor")
    trend_factor: Optional[float] = Field(None, description="Trend adjustment factor")
    
    @field_validator('date')
    @classmethod
    def validate_date_not_future(cls, v):
        """Validate that demand record date is not in the future."""
        if v.date() > date.today():
            raise ValueError('Demand record date cannot be in the future')
        return v


class DemandForecast(BaseModel):
    """
    Internal model for demand forecast processing.
    
    Contains forecast calculations, model parameters, and
    performance metrics for internal processing.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    forecast_date: datetime = Field(..., description="Forecast date")
    prediction: float = Field(..., description="Forecasted demand", ge=0)
    prediction_interval_lower: float = Field(..., description="Lower prediction interval", ge=0)
    prediction_interval_upper: float = Field(..., description="Upper prediction interval", ge=0)
    model_features: Dict[str, float] = Field(default_factory=dict, description="Features used in prediction")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    forecast_horizon: ForecastHorizon = Field(..., description="Forecast horizon period")
    seasonality_pattern: SeasonalityPattern = Field(..., description="Detected seasonality pattern")
    trend_direction: TrendDirection = Field(..., description="Detected trend direction")
    
    @field_validator('forecast_date')
    @classmethod
    def validate_forecast_date_future(cls, v):
        """Validate that forecast date is in the future."""
        if v.date() <= date.today():
            raise ValueError('Forecast date must be in the future')
        return v

    @model_validator(mode='after')
    def validate_prediction_interval(self):
        """Validate prediction interval bounds."""
        if self.prediction_interval_lower > self.prediction_interval_upper:
            raise ValueError('Lower prediction interval must be less than upper bound')
        return self


class ForecastModel(BaseModel):
    """
    Internal model for forecast model metadata.
    
    Contains model training information, performance metrics,
    and configuration parameters for model management.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    model_version: str = Field(..., description="Model version identifier")
    model_type: str = Field("LightGBM", description="Machine learning model type")
    training_data_points: int = Field(..., description="Number of training data points", gt=0)
    training_start_date: date = Field(..., description="Start date of training data")
    training_end_date: date = Field(..., description="End date of training data")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    accuracy_metrics: Dict[str, float] = Field(default_factory=dict, description="Model accuracy metrics")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    cross_validation_scores: List[float] = Field(default_factory=list, description="Cross-validation performance scores")
    created_at: datetime = Field(default_factory=datetime.now, description="Model creation timestamp")
    last_retrained: Optional[datetime] = Field(None, description="Last retraining timestamp")
    
    @model_validator(mode='after')
    def validate_training_dates(self):
        """Validate training date consistency."""
        if self.training_end_date <= self.training_start_date:
            raise ValueError('Training end date must be after start date')
        
        # Training data should not be too old (more than 2 years)
        if self.training_start_date < date.today() - timedelta(days=730):
            raise ValueError('Training data is too old (more than 2 years)')
        
        return self


# Analytics Models

class DemandAnalytics(BaseModel):
    """
    Model for demand analytics and insights.
    
    Contains statistical analysis of demand patterns,
    trends, and seasonality for business intelligence.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    analysis_period_start: date = Field(..., description="Start date of analysis period")
    analysis_period_end: date = Field(..., description="End date of analysis period")
    total_demand: int = Field(..., description="Total demand in analysis period", ge=0)
    average_daily_demand: float = Field(..., description="Average daily demand", ge=0)
    demand_variance: float = Field(..., description="Demand variance", ge=0)
    demand_std_dev: float = Field(..., description="Demand standard deviation", ge=0)
    coefficient_of_variation: float = Field(..., description="Coefficient of variation", ge=0)
    trend_direction: TrendDirection = Field(..., description="Overall trend direction")
    seasonality_pattern: SeasonalityPattern = Field(..., description="Detected seasonality pattern")
    peak_demand_date: Optional[date] = Field(None, description="Date of peak demand")
    peak_demand_value: Optional[int] = Field(None, description="Peak demand value")
    low_demand_date: Optional[date] = Field(None, description="Date of lowest demand")
    low_demand_value: Optional[int] = Field(None, description="Lowest demand value")
    outlier_count: int = Field(0, description="Number of outlier data points", ge=0)
    data_quality_score: float = Field(..., description="Data quality score (0-1)", ge=0, le=1)
    
    @model_validator(mode='after')
    def validate_analysis_period(self):
        """Validate analysis period and derived metrics."""
        if self.analysis_period_end <= self.analysis_period_start:
            raise ValueError('Analysis end date must be after start date')
        
        # Validate peak and low demand consistency
        if self.peak_demand_value is not None and self.low_demand_value is not None:
            if self.peak_demand_value < self.low_demand_value:
                raise ValueError('Peak demand value must be greater than or equal to low demand value')
        
        return self


class ForecastAccuracyMetrics(BaseModel):
    """
    Model for forecast accuracy metrics and performance evaluation.
    
    Contains various accuracy metrics used to evaluate
    forecast model performance and reliability.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    evaluation_period_start: date = Field(..., description="Start date of evaluation period")
    evaluation_period_end: date = Field(..., description="End date of evaluation period")
    forecast_horizon_weeks: int = Field(..., description="Forecast horizon in weeks", gt=0)
    total_forecasts_evaluated: int = Field(..., description="Total number of forecasts evaluated", gt=0)
    mean_absolute_error: float = Field(..., description="Mean Absolute Error (MAE)", ge=0)
    mean_absolute_percentage_error: float = Field(..., description="Mean Absolute Percentage Error (MAPE)", ge=0)
    root_mean_square_error: float = Field(..., description="Root Mean Square Error (RMSE)", ge=0)
    mean_forecast_error: float = Field(..., description="Mean Forecast Error (bias)")
    forecast_accuracy_percentage: float = Field(..., description="Overall forecast accuracy percentage", ge=0, le=100)
    directional_accuracy: float = Field(..., description="Directional accuracy percentage", ge=0, le=100)
    within_confidence_interval_percentage: float = Field(..., description="Percentage of actuals within confidence interval", ge=0, le=100)
    model_version: str = Field(..., description="Model version evaluated")
    calculated_at: datetime = Field(default_factory=datetime.now, description="Calculation timestamp")
    
    @model_validator(mode='after')
    def validate_evaluation_period(self):
        """Validate evaluation period."""
        if self.evaluation_period_end <= self.evaluation_period_start:
            raise ValueError('Evaluation end date must be after start date')
        return self


# API Request/Response Models

class DemandForecastRequest(BaseModel):
    """Request model for demand forecasting API endpoints."""
    csv_data: str = Field(..., description="CSV data as string")
    forecast_horizon_weeks: int = Field(12, description="Forecast horizon in weeks", gt=0, le=52)
    confidence_level: float = Field(0.95, description="Confidence level for intervals", gt=0, le=1)
    include_seasonality: bool = Field(True, description="Whether to include seasonality analysis")
    model_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional model parameters")


class DemandForecastResponse(BaseModel):
    """Response model for demand forecasting API endpoints."""
    success: bool = Field(..., description="Whether forecasting was successful")
    forecasts_count: int = Field(..., description="Number of forecasts generated")
    csv_output: str = Field(..., description="Forecasts as CSV string")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    model_performance: Dict[str, float] = Field(default_factory=dict, description="Model performance metrics")
    forecast_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of forecast results")


class DemandAnalyticsRequest(BaseModel):
    """Request model for demand analytics API endpoints."""
    sku: Optional[str] = Field(None, description="Filter by specific SKU")
    start_date: Optional[str] = Field(None, description="Analysis start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Analysis end date (YYYY-MM-DD)")
    include_outliers: bool = Field(False, description="Whether to include outlier analysis")
    seasonality_analysis: bool = Field(True, description="Whether to perform seasonality analysis")


class DemandAnalyticsResponse(BaseModel):
    """Response model for demand analytics API endpoints."""
    success: bool = Field(..., description="Whether analytics was successful")
    analytics_count: int = Field(..., description="Number of SKUs analyzed")
    analytics_results: List[DemandAnalytics] = Field(..., description="Analytics results")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    summary_statistics: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics across all SKUs")


# Error Models

class DemandException(Exception):
    """Base exception for demand operations."""
    pass


class InvalidForecastException(DemandException):
    """Exception raised when forecast parameters are invalid."""
    pass


class InsufficientDataException(DemandException):
    """Exception raised when insufficient data is available for forecasting."""
    pass


class ModelTrainingException(DemandException):
    """Exception raised when model training fails."""
    pass


class ForecastGenerationException(DemandException):
    """Exception raised when forecast generation fails."""
    pass


class DataQualityException(DemandException):
    """Exception raised when data quality is insufficient."""
    pass