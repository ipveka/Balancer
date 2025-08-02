"""
Inventory API Endpoints

This module implements FastAPI routes for inventory management operations
including inventory optimization, CRUD operations for inventory items,
and transaction recording with comprehensive validation and documentation.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Query
from fastapi.responses import Response
import io

from .models import (
    InventoryOptimizationRequest, InventoryOptimizationResponse,
    InventoryItemRequest, InventoryTransactionRequest, InventoryQueryParams,
    InventoryItem, InventoryTransaction, TransactionType, InventoryStatus,
    RecommendedAction, InventoryRecommendationsCSV,
    InventoryException, InsufficientInventoryException,
    InventoryItemNotFoundException, InvalidTransactionException
)
from .service import InventoryService
from utils.helpers import CSVProcessingError, DataValidationError

logger = logging.getLogger(__name__)

# Create FastAPI router with tags and metadata
router = APIRouter(
    prefix="/inventory",
    tags=["Inventory Management"],
    responses={
        404: {"description": "Resource not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Initialize service instance
inventory_service = InventoryService()


@router.get("/", summary="Inventory module health check")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the inventory module.
    
    Returns:
        Dictionary with health status and module information
    """
    return {
        "status": "healthy",
        "module": "inventory",
        "version": "1.0.0",
        "description": "Inventory management module for stock optimization and tracking",
        "supported_operations": ["optimization", "item_management", "transaction_recording"],
        "timestamp": time.time()
    }


@router.post(
    "/optimize",
    response_model=InventoryOptimizationResponse,
    summary="Optimize inventory levels",
    description="Process CSV data and generate intelligent inventory recommendations with safety stock calculations"
)
async def optimize_inventory(request: InventoryOptimizationRequest) -> InventoryOptimizationResponse:
    """
    Optimize inventory levels based on CSV input data.
    
    This endpoint processes CSV data containing inventory status information and generates
    intelligent recommendations including safety stock calculations, reorder points,
    and actionable recommendations for inventory management.
    
    Args:
        request: Inventory optimization request with CSV data
        
    Returns:
        InventoryOptimizationResponse with recommendations and processing details
        
    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    start_time = time.time()
    
    try:
        logger.info("Starting inventory optimization")
        
        # Process inventory data
        result = inventory_service.calculate_safety_stock(request.csv_data)
        csv_output = inventory_service.get_recommendations_csv(result)
        
        processing_time = time.time() - start_time
        
        # Generate optimization summary
        optimization_summary = result.optimization_summary.copy()
        optimization_summary.update({
            "processing_time_seconds": round(processing_time, 3),
            "optimization_parameters": request.optimization_params
        })
        
        response = InventoryOptimizationResponse(
            success=True,
            recommendations_count=result.recommendations_count,
            csv_output=csv_output,
            processing_time_seconds=round(processing_time, 3),
            optimization_summary=optimization_summary
        )
        
        logger.info(f"Successfully completed inventory optimization: {result.recommendations_count} recommendations generated")
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
    except Exception as e:
        logger.error(f"Unexpected error in inventory optimization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during inventory optimization"
        )


@router.post(
    "/upload",
    summary="Upload inventory CSV file",
    description="Upload a CSV file for inventory optimization and download recommendations"
)
async def upload_inventory_csv(
    file: UploadFile = File(..., description="CSV file with inventory status data"),
    optimization_params: Optional[str] = Form(None, description="JSON string with optimization parameters")
) -> Response:
    """
    Upload inventory CSV file and return optimized recommendations.
    
    This endpoint accepts file uploads and returns CSV recommendations as a downloadable file.
    
    Args:
        file: Uploaded CSV file with inventory status data
        optimization_params: Optional JSON string with optimization parameters
        
    Returns:
        CSV file response with inventory recommendations
        
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
        result = inventory_service.calculate_safety_stock(csv_content)
        csv_output = inventory_service.get_recommendations_csv(result)
        summary = result.optimization_summary
        
        # Create response with CSV content
        response = Response(
            content=csv_output,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=inventory_recommendations.csv",
                "X-Processing-Summary": str(summary)
            }
        )
        
        logger.info(f"Successfully processed inventory file upload: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing inventory file upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )


@router.post(
    "/items",
    response_model=InventoryItem,
    summary="Create inventory item",
    description="Create a new inventory item with validation and stock level calculations"
)
async def create_inventory_item(request: InventoryItemRequest) -> InventoryItem:
    """
    Create a new inventory item.
    
    Args:
        request: Inventory item creation request
        
    Returns:
        Created InventoryItem
        
    Raises:
        HTTPException: For validation errors
    """
    try:
        item_data = request.model_dump()
        inventory_item = await inventory_service.create_inventory_item(item_data)
        
        logger.info(f"Created inventory item: {inventory_item.sku}")
        return inventory_item
        
    except DataValidationError as e:
        logger.error(f"Validation error creating inventory item: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating inventory item: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create inventory item"
        )


@router.get(
    "/items",
    response_model=List[InventoryItem],
    summary="Get inventory items",
    description="Retrieve inventory items with optional filtering and pagination"
)
async def get_inventory_items(
    sku: Optional[str] = Query(None, description="Filter by SKU"),
    location: Optional[str] = Query(None, description="Filter by location"),
    status: Optional[InventoryStatus] = Query(None, description="Filter by status"),
    low_stock_only: Optional[bool] = Query(False, description="Show only items below reorder point"),
    include_inactive: Optional[bool] = Query(False, description="Include inactive items"),
    limit: Optional[int] = Query(100, description="Maximum number of results", gt=0, le=1000),
    offset: Optional[int] = Query(0, description="Number of results to skip", ge=0)
) -> List[InventoryItem]:
    """
    Get inventory items with filtering and pagination.
    
    This endpoint would typically query a database. For now, it returns
    a mock response to demonstrate the API structure.
    
    Args:
        sku: Optional SKU filter
        location: Optional location filter
        status: Optional status filter
        low_stock_only: Show only items below reorder point
        include_inactive: Include inactive items
        limit: Maximum number of results
        offset: Number of results to skip
        
    Returns:
        List of InventoryItem objects
    """
    # This is a mock implementation - in a real system, this would query a database
    logger.info(f"Retrieving inventory items with filters: sku={sku}, location={location}, status={status}")
    
    # Return empty list for now - this would be implemented with actual data storage
    return []


@router.get(
    "/items/{sku}",
    response_model=InventoryItem,
    summary="Get inventory item by SKU",
    description="Retrieve a specific inventory item by its SKU"
)
async def get_inventory_item(sku: str) -> InventoryItem:
    """
    Get a specific inventory item by SKU.
    
    Args:
        sku: Stock Keeping Unit identifier
        
    Returns:
        InventoryItem object
        
    Raises:
        HTTPException: If item not found
    """
    # This is a mock implementation - in a real system, this would query a database
    logger.info(f"Retrieving inventory item: {sku}")
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Inventory item not found: {sku}"
    )


@router.put(
    "/items/{sku}",
    response_model=InventoryItem,
    summary="Update inventory item",
    description="Update an existing inventory item"
)
async def update_inventory_item(sku: str, request: InventoryItemRequest) -> InventoryItem:
    """
    Update an existing inventory item.
    
    Args:
        sku: Stock Keeping Unit identifier
        request: Updated inventory item data
        
    Returns:
        Updated InventoryItem
        
    Raises:
        HTTPException: If item not found or validation fails
    """
    # This is a mock implementation - in a real system, this would update a database
    logger.info(f"Updating inventory item: {sku}")
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Inventory item not found: {sku}"
    )


@router.delete(
    "/items/{sku}",
    summary="Delete inventory item",
    description="Delete an inventory item (soft delete - marks as inactive)"
)
async def delete_inventory_item(sku: str) -> Dict[str, Any]:
    """
    Delete an inventory item (soft delete).
    
    Args:
        sku: Stock Keeping Unit identifier
        
    Returns:
        Confirmation message
        
    Raises:
        HTTPException: If item not found
    """
    # This is a mock implementation - in a real system, this would update a database
    logger.info(f"Deleting inventory item: {sku}")
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Inventory item not found: {sku}"
    )


@router.post(
    "/transactions",
    response_model=InventoryTransaction,
    summary="Create inventory transaction",
    description="Record a new inventory transaction (inbound, outbound, adjustment, etc.)"
)
async def create_inventory_transaction(request: InventoryTransactionRequest) -> InventoryTransaction:
    """
    Create a new inventory transaction.
    
    Args:
        request: Inventory transaction creation request
        
    Returns:
        Created InventoryTransaction
        
    Raises:
        HTTPException: For validation errors or insufficient inventory
    """
    try:
        transaction_data = request.model_dump()
        transaction = await inventory_service.create_inventory_transaction(transaction_data)
        
        logger.info(f"Created inventory transaction: {transaction.transaction_id}")
        return transaction
        
    except DataValidationError as e:
        logger.error(f"Validation error creating transaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except InsufficientInventoryException as e:
        logger.error(f"Insufficient inventory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating inventory transaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create inventory transaction"
        )


@router.get(
    "/transactions",
    response_model=List[InventoryTransaction],
    summary="Get inventory transactions",
    description="Retrieve inventory transactions with optional filtering"
)
async def get_inventory_transactions(
    sku: Optional[str] = Query(None, description="Filter by SKU"),
    transaction_type: Optional[TransactionType] = Query(None, description="Filter by transaction type"),
    location: Optional[str] = Query(None, description="Filter by location"),
    limit: Optional[int] = Query(100, description="Maximum number of results", gt=0, le=1000),
    offset: Optional[int] = Query(0, description="Number of results to skip", ge=0)
) -> List[InventoryTransaction]:
    """
    Get inventory transactions with filtering.
    
    Args:
        sku: Optional SKU filter
        transaction_type: Optional transaction type filter
        location: Optional location filter
        limit: Maximum number of results
        offset: Number of results to skip
        
    Returns:
        List of InventoryTransaction objects
    """
    # This is a mock implementation - in a real system, this would query a database
    logger.info(f"Retrieving inventory transactions with filters: sku={sku}, type={transaction_type}")
    
    # Return empty list for now - this would be implemented with actual data storage
    return []


@router.get(
    "/analytics/summary",
    summary="Get inventory analytics summary",
    description="Get summary analytics for inventory management"
)
async def get_inventory_analytics() -> Dict[str, Any]:
    """
    Get inventory analytics summary.
    
    Returns:
        Dictionary with inventory analytics and KPIs
    """
    # This is a mock implementation - in a real system, this would calculate from actual data
    return {
        "total_items": 0,
        "total_value": 0.0,
        "items_below_reorder_point": 0,
        "items_with_excess_stock": 0,
        "average_inventory_turnover": 0.0,
        "stockout_risk_items": 0,
        "last_updated": time.time()
    }


@router.get(
    "/template",
    summary="Download inventory CSV template",
    description="Download a CSV template file for inventory status data input"
)
async def download_inventory_template() -> Response:
    """
    Download CSV template for inventory status data input.
    
    Returns:
        CSV template file for inventory optimization
    """
    template_content = """sku,current_stock,lead_time_days,service_level_target,avg_weekly_demand,demand_std_dev
WIDGET-001,150,7,0.95,50.0,12.5
GADGET-002,75,14,0.90,25.0,8.0
TOOL-003,200,10,0.95,40.0,15.0
PART-004,50,21,0.85,15.0,5.0
"""
    
    return Response(
        content=template_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=inventory_template.csv"
        }
    )


@router.get(
    "/actions",
    summary="Get supported recommendation actions",
    description="Retrieve list of supported inventory recommendation actions"
)
async def get_recommendation_actions() -> Dict[str, Any]:
    """
    Get information about supported recommendation actions.
    
    Returns:
        Dictionary with supported actions and their descriptions
    """
    return {
        "supported_actions": [
            {
                "action": "REORDER",
                "description": "Stock level is below reorder point, place order soon",
                "urgency": "medium"
            },
            {
                "action": "URGENT_REORDER",
                "description": "Stock level is critically low or stockout imminent",
                "urgency": "high"
            },
            {
                "action": "SUFFICIENT_STOCK",
                "description": "Current stock levels are adequate",
                "urgency": "low"
            },
            {
                "action": "EXCESS_STOCK",
                "description": "Stock levels are higher than necessary",
                "urgency": "low"
            },
            {
                "action": "REVIEW_REQUIRED",
                "description": "Manual review needed for this item",
                "urgency": "medium"
            }
        ],
        "input_format": "CSV with columns: sku, current_stock, lead_time_days, service_level_target, avg_weekly_demand, demand_std_dev",
        "output_format": "CSV with columns: sku, safety_stock, reorder_point, current_stock, recommended_action, days_until_stockout, confidence_score"
    }


@router.get(
    "/status",
    summary="Get inventory module status",
    description="Get detailed status information about the inventory module"
)
async def get_inventory_status() -> Dict[str, Any]:
    """
    Get detailed status information about the inventory module.
    
    Returns:
        Dictionary with module status and configuration
    """
    return {
        "module": "inventory",
        "status": "operational",
        "version": "1.0.0",
        "features": {
            "inventory_optimization": True,
            "safety_stock_calculation": True,
            "reorder_point_calculation": True,
            "stockout_prediction": True,
            "item_management": True,
            "transaction_recording": True,
            "csv_processing": True,
            "file_upload": True,
            "template_download": True
        },
        "configuration": {
            "default_service_level": inventory_service.default_service_level,
            "urgent_threshold_days": inventory_service.urgent_threshold_days,
            "excess_threshold_weeks": inventory_service.excess_threshold_weeks,
            "confidence_threshold": inventory_service.confidence_threshold
        },
        "supported_formats": ["CSV"],
        "max_file_size": "10MB",
        "rate_limits": {
            "requests_per_minute": 60,
            "concurrent_optimizations": 5
        }
    }