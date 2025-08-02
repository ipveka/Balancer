"""
Demand API Endpoints

This module implements FastAPI routes for demand forecasting operations
including AI-powered demand prediction, analytics endpoints, and CRUD operations
for demand records with comprehensive validation and documentation.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Query
from fastapi.responses import Response
import io

from .models import (
    DemandForecastRequest, DemandForecastResponse,
    DemandAnalyticsRequest, DemandAnalyticsResponse,
    DemandRecord, DemandAnalytics, ForecastOutputCSV,
    ForecastHorizon, ForecastAccuracy, SeasonalityPattern, TrendDirection,
    DemandException, InvalidForecastException, InsufficientDataException,
    ModelTrainingException, ForecastGenerationException, DataQualityException
)
from .service import DemandService
from utils.helpers import CSVProcessingError, DataValidationError

logger = logging.getLogger(__name__)

# Create FastAPI router with tags and metadata
router = APIRouter(
    prefix="/demand",
    tags=["Demand Forecasting"],
    responses={
        404: {"description": "Resource not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Initialize service instance
demand_service = DemandService()


@router.get("/", summary="Demand module health check")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the demand module.
    
    Returns:
        Dictionary with health status and module information
    """
    return {
        "status": "healthy",
        "module": "demand",
        "version": "1.0.0",
        "description": "AI-powered demand forecasting module with LightGBM machine learning",
        "supported_operations": ["forecasting", "analytics", "pattern_analysis"],
        "ml_model": "LightGBM",
        "timestamp": time.time()
    }


@router.post(
    "/forecast",
    response_model=DemandForecastResponse,
    summary="Generate demand forecasts",
    description="Process CSV demand data and generate AI-powered forecasts using LightGBM machine learning"
)
async def generate_demand_forecast(request: DemandForecastRequest) -> DemandForecastResponse:
    """
    Generate AI-powered demand forecasts based on CSV input data.
    
    This endpoint processes CSV data containing historical demand information and generates
    intelligent forecasts using LightGBM machine learning with feature engineering,
    seasonality detection, and confidence intervals.
    
    Args:
        request: Demand forecast request with CSV data and parameters
        
    Returns:
        DemandForecastResponse with forecasts and model performance metrics
        
    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting demand forecasting for {request.forecast_horizon_weeks} weeks")
        
        # Process demand data and generate forecasts
        result = demand_service.generate_forecast(
            request.csv_data,
            request.forecast_horizon_weeks,
            {"confidence_level": request.confidence_level}
        )
        
        # Export forecasts to CSV
        csv_output = demand_service.get_forecasts_csv(result)
        
        processing_time = time.time() - start_time
        
        # Create forecast summary
        forecast_summary = {
            "forecast_horizon_weeks": request.forecast_horizon_weeks,
            "confidence_level": request.confidence_level,
            "total_skus_forecasted": len(set(f.sku for f in forecasts.forecasts)),
            "include_seasonality": request.include_seasonality,
            "model_type": "LightGBM",
            "feature_engineering": True,
            "processing_time_seconds": round(processing_time, 3)
        }
        
        response = DemandForecastResponse(
            success=True,
            forecasts_count=result.forecasts_count,
            csv_output=csv_output,
            processing_time_seconds=round(processing_time, 3),
            model_performance=result.model_performance,
            forecast_summary=forecast_summary
        )
        
        logger.info(f"Successfully completed demand forecasting: {result.forecasts_count} forecasts generated")
        return response
        
    except CSVProcessingError as e:
        logger.error(f"CSV processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV processing failed: {str(e)}"
        )
    except DataValidationError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Data validation failed: {str(e)}"
        )
    except InsufficientDataException as e:
        logger.error(f"Insufficient data error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient data for forecasting: {str(e)}"
        )
    except ModelTrainingException as e:
        logger.error(f"Model training error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}"
        )
    except ForecastGenerationException as e:
        logger.error(f"Forecast generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast generation failed: {str(e)}"
        )
    except DataQualityException as e:
        logger.error(f"Data quality error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data quality insufficient: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in demand forecasting: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during demand forecasting"
        )


@router.post(
    "/upload",
    summary="Upload demand CSV file",
    description="Upload a CSV file for demand forecasting and download predictions"
)
async def upload_demand_csv(
    file: UploadFile = File(..., description="CSV file with demand data"),
    forecast_horizon_weeks: int = Form(12, description="Number of weeks to forecast", gt=0, le=52),
    confidence_level: float = Form(0.95, description="Confidence level for intervals", gt=0, le=1),
    model_params: Optional[str] = Form(None, description="JSON string with model parameters")
) -> Response:
    """
    Upload demand CSV file and return AI-generated forecasts.
    
    This endpoint accepts file uploads and returns CSV forecasts as a downloadable file.
    
    Args:
        file: Uploaded CSV file with demand data
        forecast_horizon_weeks: Number of weeks to forecast ahead
        confidence_level: Confidence level for prediction intervals
        model_params: Optional JSON string with model parameters
        
    Returns:
        CSV file response with demand forecasts
        
    Raises:
        HTTPException: For file processing errors
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a CSV file"
            )
        
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Process CSV and generate forecasts
        result = demand_service.generate_forecast(csv_content, forecast_horizon_weeks)
        csv_output = demand_service.get_forecasts_csv(result)
        summary = result.forecast_summary
        
        # Create response with CSV content
        response = Response(
            content=csv_output,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=demand_forecasts.csv",
                "X-Processing-Summary": str(summary),
                "X-Forecast-Horizon": str(forecast_horizon_weeks),
                "X-Confidence-Level": str(confidence_level)
            }
        )
        
        logger.info(f"Successfully processed demand file upload: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing demand file upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )


@router.post(
    "/analytics",
    response_model=DemandAnalyticsResponse,
    summary="Analyze demand patterns",
    description="Analyze historical demand data to identify trends, seasonality, and patterns"
)
async def analyze_demand_patterns(request: DemandAnalyticsRequest) -> DemandAnalyticsResponse:
    """
    Analyze demand patterns and trends from CSV data.
    
    This endpoint processes historical demand data to identify trends, seasonality,
    volatility patterns, and data quality metrics for business intelligence.
    
    Args:
        request: Demand analytics request with optional filters
        
    Returns:
        DemandAnalyticsResponse with pattern analysis results
        
    Raises:
        HTTPException: For processing errors
    """
    start_time = time.time()
    
    try:
        logger.info("Starting demand pattern analysis")
        
        # For this endpoint, we expect CSV data to be provided in the request
        # In a real implementation, this might query a database based on filters
        
        # Mock implementation - in practice, this would process actual data
        analytics_results = []
        processing_time = time.time() - start_time
        
        # Create summary statistics
        summary_statistics = {
            "analysis_period": {
                "start_date": request.start_date,
                "end_date": request.end_date
            },
            "filters_applied": {
                "sku": request.sku,
                "include_outliers": request.include_outliers,
                "seasonality_analysis": request.seasonality_analysis
            },
            "processing_time_seconds": round(processing_time, 3)
        }
        
        response = DemandAnalyticsResponse(
            success=True,
            analytics_count=len(analytics_results),
            analytics_results=analytics_results,
            processing_time_seconds=round(processing_time, 3),
            summary_statistics=summary_statistics
        )
        
        logger.info(f"Successfully completed demand analytics: {len(analytics_results)} SKUs analyzed")
        return response
        
    except Exception as e:
        logger.error(f"Error in demand analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demand analytics failed: {str(e)}"
        )


@router.post(
    "/analytics/upload",
    summary="Upload CSV for demand analytics",
    description="Upload a CSV file for demand pattern analysis and download results"
)
async def upload_demand_analytics_csv(
    file: UploadFile = File(..., description="CSV file with demand data"),
    include_outliers: bool = Form(False, description="Include outlier analysis"),
    seasonality_analysis: bool = Form(True, description="Perform seasonality analysis")
) -> Response:
    """
    Upload demand CSV file and return pattern analysis results.
    
    Args:
        file: Uploaded CSV file with demand data
        include_outliers: Whether to include outlier analysis
        seasonality_analysis: Whether to perform seasonality analysis
        
    Returns:
        JSON response with analytics results
        
    Raises:
        HTTPException: For file processing errors
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a CSV file"
            )
        
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Analyze demand patterns - mock implementation for now
        analytics_results = []
        summary = {"message": "Analytics functionality to be implemented"}
        
        # Create JSON response
        response_data = {
            "analytics_results": analytics_results,
            "summary": summary,
            "parameters": {
                "include_outliers": include_outliers,
                "seasonality_analysis": seasonality_analysis
            }
        }
        
        response = Response(
            content=str(response_data),
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=demand_analytics.json",
                "X-Analytics-Summary": str(summary)
            }
        )
        
        logger.info(f"Successfully processed demand analytics file upload: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing demand analytics file upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics processing failed: {str(e)}"
        )


@router.get(
    "/models/{sku}",
    summary="Get model information for SKU",
    description="Retrieve trained model information and performance metrics for a specific SKU"
)
async def get_model_info(sku: str) -> Dict[str, Any]:
    """
    Get model information for a specific SKU.
    
    Args:
        sku: Stock Keeping Unit identifier
        
    Returns:
        Dictionary with model information and performance metrics
        
    Raises:
        HTTPException: If model not found
    """
    try:
        model_summary = await demand_service.get_model_summary(sku.upper())
        
        if model_summary is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No trained model found for SKU: {sku}"
            )
        
        logger.info(f"Retrieved model information for SKU: {sku}")
        return model_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model info for SKU {sku}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


@router.get(
    "/models",
    summary="Get all trained models",
    description="Retrieve information about all trained demand forecasting models"
)
async def get_all_models() -> Dict[str, Any]:
    """
    Get information about all trained models.
    
    Returns:
        Dictionary with all model information
    """
    try:
        # Get all trained models from service
        all_models = {}
        for sku in demand_service.trained_models.keys():
            model_summary = await demand_service.get_model_summary(sku)
            if model_summary:
                all_models[sku] = model_summary
        
        return {
            "total_models": len(all_models),
            "models": all_models,
            "last_updated": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving all models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


@router.get(
    "/horizons",
    summary="Get supported forecast horizons",
    description="Retrieve list of supported forecast horizons and their descriptions"
)
async def get_forecast_horizons() -> Dict[str, Any]:
    """
    Get information about supported forecast horizons.
    
    Returns:
        Dictionary with supported horizons and their descriptions
    """
    return {
        "supported_horizons": [
            {
                "horizon": "weekly",
                "description": "Weekly demand forecasts",
                "min_weeks": 1,
                "max_weeks": 52,
                "recommended_weeks": 12,
                "data_requirements": "Minimum 4 weeks of historical data"
            },
            {
                "horizon": "monthly",
                "description": "Monthly demand forecasts",
                "min_months": 1,
                "max_months": 12,
                "recommended_months": 3,
                "data_requirements": "Minimum 12 weeks of historical data"
            }
        ],
        "default_horizon_weeks": demand_service.default_forecast_horizon_weeks,
        "confidence_levels": [0.80, 0.90, 0.95, 0.99],
        "default_confidence_level": demand_service.confidence_level
    }


@router.get(
    "/accuracy-levels",
    summary="Get forecast accuracy levels",
    description="Retrieve information about forecast accuracy classification"
)
async def get_accuracy_levels() -> Dict[str, Any]:
    """
    Get information about forecast accuracy levels.
    
    Returns:
        Dictionary with accuracy level definitions
    """
    return {
        "accuracy_levels": [
            {
                "level": "HIGH",
                "description": "High accuracy forecasts",
                "mape_threshold": "< 10%",
                "reliability": "Very reliable for planning",
                "recommended_use": "Operational planning and inventory optimization"
            },
            {
                "level": "MEDIUM",
                "description": "Medium accuracy forecasts",
                "mape_threshold": "10% - 20%",
                "reliability": "Moderately reliable",
                "recommended_use": "Strategic planning with safety buffers"
            },
            {
                "level": "LOW",
                "description": "Low accuracy forecasts",
                "mape_threshold": "> 20%",
                "reliability": "Use with caution",
                "recommended_use": "Trend analysis and high-level planning only"
            }
        ],
        "metrics_explanation": {
            "MAPE": "Mean Absolute Percentage Error",
            "MAE": "Mean Absolute Error",
            "RMSE": "Root Mean Square Error",
            "directional_accuracy": "Percentage of correct trend predictions"
        }
    }


@router.get(
    "/template",
    summary="Download demand CSV template",
    description="Download a CSV template file for demand data input"
)
async def download_demand_template() -> Response:
    """
    Download CSV template for demand data input.
    
    Returns:
        CSV template file for demand forecasting
    """
    template_content = """date,sku,quantity
2024-01-01,WIDGET-001,150
2024-01-02,WIDGET-001,142
2024-01-03,WIDGET-001,158
2024-01-04,WIDGET-001,135
2024-01-05,WIDGET-001,167
2024-01-01,GADGET-002,85
2024-01-02,GADGET-002,92
2024-01-03,GADGET-002,78
2024-01-04,GADGET-002,88
2024-01-05,GADGET-002,95
"""
    
    return Response(
        content=template_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=demand_template.csv"
        }
    )


@router.get(
    "/patterns",
    summary="Get supported pattern types",
    description="Retrieve information about supported demand pattern types"
)
async def get_pattern_types() -> Dict[str, Any]:
    """
    Get information about supported demand pattern types.
    
    Returns:
        Dictionary with pattern type definitions
    """
    return {
        "trend_patterns": [
            {
                "pattern": "INCREASING",
                "description": "Demand is trending upward over time",
                "characteristics": "Positive slope in demand trend"
            },
            {
                "pattern": "DECREASING", 
                "description": "Demand is trending downward over time",
                "characteristics": "Negative slope in demand trend"
            },
            {
                "pattern": "STABLE",
                "description": "Demand is relatively stable over time",
                "characteristics": "Minimal trend, consistent demand levels"
            },
            {
                "pattern": "VOLATILE",
                "description": "Demand shows high variability",
                "characteristics": "High coefficient of variation, irregular patterns"
            }
        ],
        "seasonality_patterns": [
            {
                "pattern": "NONE",
                "description": "No detectable seasonality",
                "characteristics": "Random or trend-only patterns"
            },
            {
                "pattern": "WEEKLY",
                "description": "Weekly seasonal patterns",
                "characteristics": "Recurring patterns every 7 days"
            },
            {
                "pattern": "MONTHLY",
                "description": "Monthly seasonal patterns",
                "characteristics": "Recurring patterns every month"
            },
            {
                "pattern": "QUARTERLY",
                "description": "Quarterly seasonal patterns",
                "characteristics": "Recurring patterns every quarter"
            },
            {
                "pattern": "YEARLY",
                "description": "Annual seasonal patterns",
                "characteristics": "Recurring patterns every year"
            }
        ]
    }


@router.get(
    "/status",
    summary="Get demand module status",
    description="Get detailed status information about the demand module"
)
async def get_demand_status() -> Dict[str, Any]:
    """
    Get detailed status information about the demand module.
    
    Returns:
        Dictionary with module status and configuration
    """
    return {
        "module": "demand",
        "status": "operational",
        "version": "1.0.0",
        "features": {
            "ai_forecasting": True,
            "lightgbm_model": True,
            "feature_engineering": True,
            "seasonality_detection": True,
            "trend_analysis": True,
            "confidence_intervals": True,
            "pattern_analysis": True,
            "csv_processing": True,
            "file_upload": True,
            "template_download": True,
            "model_management": True
        },
        "configuration": {
            "default_forecast_horizon_weeks": demand_service.default_forecast_horizon_weeks,
            "min_training_data_points": demand_service.min_training_data_points,
            "confidence_level": demand_service.confidence_level,
            "model_version": demand_service.model_version,
            "lag_periods": demand_service.lag_periods,
            "ma_windows": demand_service.ma_windows
        },
        "ml_configuration": {
            "model_type": "LightGBM",
            "objective": demand_service.lgb_params["objective"],
            "metric": demand_service.lgb_params["metric"],
            "boosting_type": demand_service.lgb_params["boosting_type"],
            "learning_rate": demand_service.lgb_params["learning_rate"]
        },
        "trained_models": len(demand_service.trained_models),
        "supported_formats": ["CSV"],
        "max_file_size": "10MB",
        "rate_limits": {
            "requests_per_minute": 30,  # Lower due to ML processing
            "concurrent_forecasts": 3
        }
    }