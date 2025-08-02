"""
Configuration management for the Balancer platform.

This module provides centralized configuration using Pydantic Settings
with support for environment variables and AI model parameters.
"""

from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class AIConfig(BaseSettings):
    """AI and Machine Learning configuration parameters."""
    
    # Demand Forecasting Configuration
    forecast_frequency: str = Field(default="weekly", description="Frequency of demand forecasting")
    forecast_horizon: int = Field(default=12, description="Forecast horizon in weeks")
    lightgbm_params: Dict[str, Any] = Field(
        default={
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        },
        description="LightGBM model parameters"
    )
    
    # Inventory Management Configuration
    default_service_level: float = Field(default=0.95, description="Default service level target")
    safety_stock_multiplier: float = Field(default=1.65, description="Z-score for 95% service level")
    seasonality_adjustment_factor: float = Field(default=0.2, description="Seasonality adjustment factor")
    
    # Supply Optimization Configuration
    lead_time_buffer_days: int = Field(default=2, description="Buffer days for lead time calculations")
    batch_size_optimization: bool = Field(default=True, description="Enable batch size optimization")
    capacity_utilization_target: float = Field(default=0.85, description="Target capacity utilization")
    
    # Distribution VRP Configuration
    vehicle_capacity_default: int = Field(default=1000, description="Default vehicle capacity")
    max_route_distance: float = Field(default=500.0, description="Maximum route distance in km")
    vrp_algorithm: str = Field(default="greedy", description="VRP algorithm: greedy or nearest_neighbor")
    optimization_iterations: int = Field(default=100, description="Number of optimization iterations")
    transport_capacity_limit: float = Field(default=10000.0, description="Transport capacity limit")


class DatabaseConfig(BaseSettings):
    """Database configuration (for future expansion)."""
    
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    database_timeout: int = Field(default=30, description="Database connection timeout in seconds")


class APIConfig(BaseSettings):
    """API configuration settings."""
    
    title: str = Field(default="Balancer API", description="API title")
    description: str = Field(default="AI-powered supply chain optimization platform", description="API description")
    version: str = Field(default="1.0.0", description="API version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # CORS Configuration
    cors_origins: list = Field(default=["*"], description="CORS allowed origins")
    cors_methods: list = Field(default=["*"], description="CORS allowed methods")
    cors_headers: list = Field(default=["*"], description="CORS allowed headers")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    # Environment
    environment: str = Field(default="development", description="Application environment")
    
    # AI Configuration
    ai: AIConfig = Field(default_factory=AIConfig)
    
    # Database Configuration
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # API Configuration
    api: APIConfig = Field(default_factory=APIConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def validate_ai_config() -> bool:
    """Validate AI configuration parameters."""
    ai_config = settings.ai
    
    # Validate service level
    if not 0.5 <= ai_config.default_service_level <= 0.99:
        raise ValueError("Service level must be between 0.5 and 0.99")
    
    # Validate forecast horizon
    if ai_config.forecast_horizon < 1:
        raise ValueError("Forecast horizon must be at least 1 week")
    
    # Validate capacity utilization target
    if not 0.1 <= ai_config.capacity_utilization_target <= 1.0:
        raise ValueError("Capacity utilization target must be between 0.1 and 1.0")
    
    # Validate VRP algorithm
    if ai_config.vrp_algorithm not in ["greedy", "nearest_neighbor"]:
        raise ValueError("VRP algorithm must be 'greedy' or 'nearest_neighbor'")
    
    return True


def get_lightgbm_params() -> Dict[str, Any]:
    """Get LightGBM parameters for model training."""
    return settings.ai.lightgbm_params.copy()


def get_service_level() -> float:
    """Get the configured service level."""
    return settings.ai.default_service_level


def get_forecast_config() -> Dict[str, Any]:
    """Get demand forecasting configuration."""
    return {
        "frequency": settings.ai.forecast_frequency,
        "horizon": settings.ai.forecast_horizon,
        "seasonality_factor": settings.ai.seasonality_adjustment_factor
    }


def get_supply_config() -> Dict[str, Any]:
    """Get supply optimization configuration."""
    return {
        "lead_time_buffer": settings.ai.lead_time_buffer_days,
        "batch_optimization": settings.ai.batch_size_optimization,
        "capacity_target": settings.ai.capacity_utilization_target
    }


def get_distribution_config() -> Dict[str, Any]:
    """Get distribution optimization configuration."""
    return {
        "default_capacity": settings.ai.vehicle_capacity_default,
        "max_distance": settings.ai.max_route_distance,
        "algorithm": settings.ai.vrp_algorithm,
        "iterations": settings.ai.optimization_iterations,
        "transport_limit": settings.ai.transport_capacity_limit
    }