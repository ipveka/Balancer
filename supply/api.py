"""
Supply API Endpoints

This module implements FastAPI routes for supply management operations
including procurement and manufacturing optimization with comprehensive
request/response validation and OpenAPI documentation.
"""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import Response
import io

from .models import (
    SupplyOptimizationRequest, SupplyOptimizationResponse,
    SupplyMode, ProcurementRecommendationsCSV, ManufacturingRecommendationsCSV,
    SupplyChainException, OptimizationFailedException
)
from .service import SupplyService
from utils.helpers import CSVProcessingError, DataValidationError

logger = logging.getLogger(__name__)

# Create FastAPI router with tags and metadata
router = APIRouter(
    prefix="/supply",
    tags=["Supply Management"],
    responses={
        404: {"description": "Resource not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Initialize service instance
supply_service = SupplyService()


@router.get("/", summary="Supply module health check")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the supply module.
    
    Returns:
        Dictionary with health status and module information
    """
    return {
        "status": "healthy",
        "module": "supply",
        "version": "1.0.0",
        "description": "Supply management module for procurement and manufacturing optimization",
        "supported_modes": ["procurement", "manufacturing"],
        "timestamp": time.time()
    }


@router.post(
    "/optimize",
    response_model=SupplyOptimizationResponse,
    summary="Optimize supply operations",
    description="Process CSV data and generate optimized supply recommendations for procurement or manufacturing mode"
)
async def optimize_supply(request: SupplyOptimizationRequest) -> SupplyOptimizationResponse:
    """
    Optimize supply operations based on CSV input data.
    
    This endpoint processes CSV data containing supply requirements and generates
    optimized recommendations for either procurement or manufacturing operations.
    
    Args:
        request: Supply optimization request with mode and CSV data
        
    Returns:
        SupplyOptimizationResponse with recommendations and processing details
        
    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting supply optimization in {request.mode} mode")
        
        if request.mode == SupplyMode.PROCUREMENT:
            result = supply_service.optimize_procurement(request.csv_data, request.optimization_params)
            csv_output = supply_service.get_recommendations_csv(result)
            recommendations_count = result.recommendations_count
            
        elif request.mode == SupplyMode.MANUFACTURING:
            result = supply_service.optimize_manufacturing(request.csv_data, request.optimization_params)
            csv_output = supply_service.get_recommendations_csv(result)
            recommendations_count = result.recommendations_count
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported supply mode: {request.mode}"
            )
        
        processing_time = time.time() - start_time
        
        # Create optimization summary
        optimization_summary = {
            "mode": request.mode.value,
            "total_items_processed": recommendations_count,
            "processing_time_seconds": round(processing_time, 3),
            "optimization_parameters": request.optimization_params,
            "success_rate": 100.0  # Assume 100% success if we reach here
        }
        
        response = SupplyOptimizationResponse(
            success=True,
            mode=request.mode,
            recommendations_count=recommendations_count,
            csv_output=csv_output,
            processing_time_seconds=round(processing_time, 3),
            optimization_summary=optimization_summary
        )
        
        logger.info(f"Successfully completed supply optimization: {recommendations_count} recommendations generated")
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
    except OptimizationFailedException as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in supply optimization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during supply optimization"
        )


@router.post(
    "/procurement/optimize",
    response_model=SupplyOptimizationResponse,
    summary="Optimize procurement operations",
    description="Process procurement CSV data and generate optimized purchase recommendations"
)
async def optimize_procurement(request: SupplyOptimizationRequest) -> SupplyOptimizationResponse:
    """
    Optimize procurement operations specifically.
    
    This endpoint is a convenience wrapper that forces procurement mode
    and processes CSV data to generate purchase order recommendations.
    
    Args:
        request: Supply optimization request (mode will be overridden to procurement)
        
    Returns:
        SupplyOptimizationResponse with procurement recommendations
    """
    # Override mode to ensure procurement
    request.mode = SupplyMode.PROCUREMENT
    return await optimize_supply(request)


@router.post(
    "/manufacturing/optimize",
    response_model=SupplyOptimizationResponse,
    summary="Optimize manufacturing operations",
    description="Process manufacturing CSV data and generate optimized production recommendations"
)
async def optimize_manufacturing(request: SupplyOptimizationRequest) -> SupplyOptimizationResponse:
    """
    Optimize manufacturing operations specifically.
    
    This endpoint is a convenience wrapper that forces manufacturing mode
    and processes CSV data to generate production batch recommendations.
    
    Args:
        request: Supply optimization request (mode will be overridden to manufacturing)
        
    Returns:
        SupplyOptimizationResponse with manufacturing recommendations
    """
    # Override mode to ensure manufacturing
    request.mode = SupplyMode.MANUFACTURING
    return await optimize_supply(request)


@router.post(
    "/procurement/upload",
    summary="Upload procurement CSV file",
    description="Upload a CSV file for procurement optimization and download recommendations"
)
async def upload_procurement_csv(
    file: UploadFile = File(..., description="CSV file with procurement data"),
    optimization_params: Optional[str] = Form(None, description="JSON string with optimization parameters")
) -> Response:
    """
    Upload procurement CSV file and return optimized recommendations.
    
    This endpoint accepts file uploads and returns CSV recommendations as a downloadable file.
    
    Args:
        file: Uploaded CSV file with procurement data
        optimization_params: Optional JSON string with optimization parameters
        
    Returns:
        CSV file response with procurement recommendations
        
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
        
        # Process CSV
        result = supply_service.optimize_procurement(csv_content)
        csv_output = supply_service.get_recommendations_csv(result)
        summary = result.optimization_summary
        
        # Create response with CSV content
        response = Response(
            content=csv_output,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=procurement_recommendations.csv",
                "X-Processing-Summary": str(summary)
            }
        )
        
        logger.info(f"Successfully processed procurement file upload: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing procurement file upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )


@router.post(
    "/manufacturing/upload",
    summary="Upload manufacturing CSV file",
    description="Upload a CSV file for manufacturing optimization and download recommendations"
)
async def upload_manufacturing_csv(
    file: UploadFile = File(..., description="CSV file with manufacturing data"),
    optimization_params: Optional[str] = Form(None, description="JSON string with optimization parameters")
) -> Response:
    """
    Upload manufacturing CSV file and return optimized recommendations.
    
    This endpoint accepts file uploads and returns CSV recommendations as a downloadable file.
    
    Args:
        file: Uploaded CSV file with manufacturing data
        optimization_params: Optional JSON string with optimization parameters
        
    Returns:
        CSV file response with manufacturing recommendations
        
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
        
        # Process CSV
        result = supply_service.optimize_manufacturing(csv_content)
        csv_output = supply_service.get_recommendations_csv(result)
        summary = result.optimization_summary
        
        # Create response with CSV content
        response = Response(
            content=csv_output,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=manufacturing_recommendations.csv",
                "X-Processing-Summary": str(summary)
            }
        )
        
        logger.info(f"Successfully processed manufacturing file upload: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing manufacturing file upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )


@router.get(
    "/modes",
    summary="Get supported supply modes",
    description="Retrieve list of supported supply operation modes and their descriptions"
)
async def get_supply_modes() -> Dict[str, Any]:
    """
    Get information about supported supply modes.
    
    Returns:
        Dictionary with supported modes and their descriptions
    """
    return {
        "supported_modes": [
            {
                "mode": "procurement",
                "description": "Optimize purchase orders and supplier management",
                "input_format": "CSV with columns: sku, current_inventory, forecast_demand_4weeks, safety_stock, min_order_qty, supplier_id, unit_cost",
                "output_format": "CSV with columns: sku, recommended_quantity, supplier_id, order_date, expected_delivery, total_cost, recommendation_action, confidence_score"
            },
            {
                "mode": "manufacturing",
                "description": "Optimize production batches and manufacturing schedules",
                "input_format": "CSV with columns: sku, current_inventory, forecast_demand_4weeks, safety_stock, batch_size, production_time_days, unit_cost",
                "output_format": "CSV with columns: sku, recommended_batch_qty, production_start_date, production_complete_date, total_cost, recommendation_action, confidence_score"
            }
        ],
        "recommendation_actions": [
            "ORDER", "PRODUCE", "NO_ACTION", "URGENT_ORDER", "URGENT_PRODUCE"
        ]
    }


@router.get(
    "/templates/procurement",
    summary="Download procurement CSV template",
    description="Download a CSV template file for procurement data input"
)
async def download_procurement_template() -> Response:
    """
    Download CSV template for procurement data input.
    
    Returns:
        CSV template file for procurement operations
    """
    template_content = """sku,current_inventory,forecast_demand_4weeks,safety_stock,min_order_qty,supplier_id,unit_cost
WIDGET-001,100,500,50,100,SUP-001,10.50
GADGET-002,25,200,30,50,SUP-002,25.75
TOOL-003,0,150,20,25,SUP-001,15.25
"""
    
    return Response(
        content=template_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=procurement_template.csv"
        }
    )


@router.get(
    "/templates/manufacturing",
    summary="Download manufacturing CSV template",
    description="Download a CSV template file for manufacturing data input"
)
async def download_manufacturing_template() -> Response:
    """
    Download CSV template for manufacturing data input.
    
    Returns:
        CSV template file for manufacturing operations
    """
    template_content = """sku,current_inventory,forecast_demand_4weeks,safety_stock,batch_size,production_time_days,unit_cost
WIDGET-001,100,500,50,200,5,8.50
GADGET-002,25,200,30,100,3,18.75
TOOL-003,0,150,20,50,7,12.25
"""
    
    return Response(
        content=template_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=manufacturing_template.csv"
        }
    )


@router.get(
    "/status",
    summary="Get supply module status",
    description="Get detailed status information about the supply module"
)
async def get_supply_status() -> Dict[str, Any]:
    """
    Get detailed status information about the supply module.
    
    Returns:
        Dictionary with module status and configuration
    """
    return {
        "module": "supply",
        "status": "operational",
        "version": "1.0.0",
        "features": {
            "procurement_optimization": True,
            "manufacturing_optimization": True,
            "csv_processing": True,
            "file_upload": True,
            "template_download": True
        },
        "configuration": {
            "default_lead_time_days": supply_service.default_lead_time_days,
            "default_safety_buffer": supply_service.default_safety_buffer,
            "optimization_confidence_threshold": supply_service.optimization_confidence_threshold
        },
        "supported_formats": ["CSV"],
        "max_file_size": "10MB",
        "rate_limits": {
            "requests_per_minute": 60,
            "concurrent_optimizations": 5
        }
    }