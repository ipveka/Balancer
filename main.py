"""
Main FastAPI Application for Balancer Platform

This module serves as the entry point for the Balancer AI-powered supply chain
optimization platform. It initializes the FastAPI application, includes all
domain module routers, and sets up global middleware, exception handling,
and health check endpoints.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

# Import configuration
from config import settings, validate_ai_config

# Import domain module routers
from supply.api import router as supply_router
from inventory.api import router as inventory_router
from demand.api import router as demand_router
from distribution.api import router as distribution_router

# Import domain-specific exceptions
from supply.models import (
    SupplyChainException, SupplierNotFoundException, 
    InvalidSupplyModeException, OptimizationFailedException
)
from inventory.models import (
    InventoryException, InsufficientInventoryException,
    InventoryItemNotFoundException, InvalidTransactionException,
    StockCalculationException
)
from demand.models import (
    DemandException, InvalidForecastException, InsufficientDataException,
    ModelTrainingException, ForecastGenerationException, DataQualityException
)
from utils.helpers import CSVProcessingError, DataValidationError
from utils.ml_utils import MLUtilsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("balancer.log") if settings.environment != "development" else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    This function handles application initialization and cleanup tasks
    including configuration validation and resource management.
    """
    # Startup
    logger.info("Starting Balancer platform...")
    
    try:
        # Validate AI configuration
        validate_ai_config()
        logger.info("AI configuration validated successfully")
        
        # Log configuration summary
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.api.debug}")
        logger.info(f"Forecast frequency: {settings.ai.forecast_frequency}")
        logger.info(f"Service level: {settings.ai.default_service_level}")
        logger.info(f"VRP algorithm: {settings.ai.vrp_algorithm}")
        
        # Initialize any required resources here
        # (database connections, ML models, etc.)
        
        logger.info("Balancer platform started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start Balancer platform: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Balancer platform...")
    
    # Cleanup resources here
    # (close database connections, save models, etc.)
    
    logger.info("Balancer platform shutdown complete")


# Create FastAPI application with metadata
app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    debug=settings.api.debug,
    lifespan=lifespan,
    docs_url="/docs" if settings.api.debug else None,
    redoc_url="/redoc" if settings.api.debug else None,
    openapi_url="/openapi.json" if settings.api.debug else None,
    contact={
        "name": "Balancer Support",
        "email": "support@balancer.ai",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.balancer.ai",
            "description": "Production server"
        }
    ]
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=settings.api.cors_methods,
    allow_headers=settings.api.cors_headers,
)


# Add trusted host middleware for production
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.balancer.ai", "*.balancer.ai"]
    )


# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """
    Middleware for adding security headers to responses.
    
    Adds standard security headers to protect against common vulnerabilities.
    """
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Add CSP header for production
    if settings.environment == "production":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
    
    return response


# Request ID generation middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Middleware for generating unique request IDs for tracking.
    
    Generates a unique request ID for each incoming request to enable
    request tracking across logs and error responses.
    """
    import uuid
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Process request
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response


# Request/Response logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Middleware for request/response logging and performance monitoring.
    
    Logs all incoming requests with timing information and response status.
    """
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log request
    logger.info(
        f"Request [{request_id}]: {request.method} {request.url.path} - "
        f"Client: {request.client.host} - "
        f"User-Agent: {request.headers.get('user-agent', 'unknown')}"
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        f"Response [{request_id}]: {response.status_code} - "
        f"Time: {process_time:.3f}s - "
        f"Path: {request.url.path} - "
        f"Size: {response.headers.get('content-length', 'unknown')} bytes"
    )
    
    # Add performance headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Server-Version"] = settings.api.version
    
    return response


# Include domain module routers with proper prefixes and tags
app.include_router(
    supply_router,
    prefix="/api/v1",
    tags=["Supply Management"]
)

app.include_router(
    inventory_router,
    prefix="/api/v1",
    tags=["Inventory Management"]
)

app.include_router(
    demand_router,
    prefix="/api/v1",
    tags=["Demand Forecasting"]
)

app.include_router(
    distribution_router,
    prefix="/api/v1",
    tags=["Distribution Planning"]
)


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Global HTTP exception handler with structured error responses.
    
    Provides consistent error response format across all endpoints.
    """
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Request validation exception handler with detailed field-level errors.
    
    Provides detailed validation error information for debugging.
    """
    logger.error(f"Validation Error: {exc.errors()} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.errors(),
            "path": str(request.url.path),
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Starlette HTTP exception handler for framework-level errors.
    
    Handles low-level HTTP exceptions from the Starlette framework.
    """
    logger.error(f"Starlette Exception: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "FRAMEWORK_ERROR",
            "message": str(exc.detail),
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )


# Domain-specific exception handlers

@app.exception_handler(SupplyChainException)
async def supply_chain_exception_handler(request: Request, exc: SupplyChainException):
    """Handle supply chain specific exceptions."""
    logger.error(f"Supply Chain Error: {str(exc)} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "SUPPLY_CHAIN_ERROR",
            "message": str(exc),
            "module": "supply",
            "path": str(request.url.path),
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(InventoryException)
async def inventory_exception_handler(request: Request, exc: InventoryException):
    """Handle inventory specific exceptions."""
    logger.error(f"Inventory Error: {str(exc)} - Path: {request.url.path}")
    
    # Determine appropriate status code based on exception type
    status_code = status.HTTP_400_BAD_REQUEST
    if isinstance(exc, InventoryItemNotFoundException):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, InvalidTransactionException):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": "INVENTORY_ERROR",
            "message": str(exc),
            "module": "inventory",
            "exception_type": exc.__class__.__name__,
            "path": str(request.url.path),
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(DemandException)
async def demand_exception_handler(request: Request, exc: DemandException):
    """Handle demand forecasting specific exceptions."""
    logger.error(f"Demand Error: {str(exc)} - Path: {request.url.path}")
    
    # Determine appropriate status code based on exception type
    status_code = status.HTTP_400_BAD_REQUEST
    if isinstance(exc, (ModelTrainingException, ForecastGenerationException)):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    elif isinstance(exc, InvalidForecastException):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": "DEMAND_ERROR",
            "message": str(exc),
            "module": "demand",
            "exception_type": exc.__class__.__name__,
            "path": str(request.url.path),
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(CSVProcessingError)
async def csv_processing_exception_handler(request: Request, exc: CSVProcessingError):
    """Handle CSV processing specific exceptions."""
    logger.error(f"CSV Processing Error: {str(exc)} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "CSV_PROCESSING_ERROR",
            "message": str(exc),
            "path": str(request.url.path),
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(DataValidationError)
async def data_validation_exception_handler(request: Request, exc: DataValidationError):
    """Handle data validation specific exceptions."""
    logger.error(f"Data Validation Error: {str(exc)} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "DATA_VALIDATION_ERROR",
            "message": str(exc),
            "path": str(request.url.path),
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(MLUtilsError)
async def ml_utils_exception_handler(request: Request, exc: MLUtilsError):
    """Handle ML utilities specific exceptions."""
    logger.error(f"ML Utils Error: {str(exc)} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "ML_UTILS_ERROR",
            "message": str(exc),
            "path": str(request.url.path),
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    General exception handler for unexpected errors.
    
    Catches all unhandled exceptions and provides a consistent error response.
    """
    logger.error(f"Unexpected Error: {str(exc)} - Path: {request.url.path}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred" if settings.environment == "production" else str(exc),
            "status_code": 500,
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )


# Health check endpoints
@app.get("/", tags=["Health"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint with basic platform information.
    
    Returns:
        Dictionary with platform information and status
    """
    return {
        "name": "Balancer Platform",
        "description": "AI-powered supply chain optimization platform",
        "version": settings.api.version,
        "status": "operational",
        "environment": settings.environment,
        "modules": ["supply", "inventory", "demand", "distribution"],
        "documentation": "/docs" if settings.api.debug else "Contact support for API documentation",
        "timestamp": time.time()
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    
    Returns:
        Dictionary with detailed health status of all modules
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.api.version,
        "environment": settings.environment,
        "modules": {
            "supply": "operational",
            "inventory": "operational", 
            "demand": "operational",
            "distribution": "operational"
        },
        "configuration": {
            "ai_enabled": True,
            "forecast_frequency": settings.ai.forecast_frequency,
            "service_level": settings.ai.default_service_level,
            "vrp_algorithm": settings.ai.vrp_algorithm,
            "debug_mode": settings.api.debug
        },
        "system": {
            "uptime_seconds": time.time(),  # This would be actual uptime in production
            "memory_usage": "N/A",  # Would be actual memory usage in production
            "cpu_usage": "N/A"  # Would be actual CPU usage in production
        }
    }


@app.get("/api/v1/status", tags=["Health"])
async def api_status() -> Dict[str, Any]:
    """
    API status endpoint with module-specific information.
    
    Returns:
        Dictionary with API status and module availability
    """
    return {
        "api_version": "v1",
        "status": "operational",
        "timestamp": time.time(),
        "endpoints": {
            "supply": "/api/v1/supply",
            "inventory": "/api/v1/inventory",
            "demand": "/api/v1/demand",
            "distribution": "/api/v1/distribution"
        },
        "features": {
            "ai_forecasting": True,
            "route_optimization": True,
            "inventory_optimization": True,
            "supply_planning": True,
            "csv_processing": True,
            "file_uploads": True
        },
        "rate_limits": {
            "requests_per_minute": settings.api.rate_limit_requests,
            "window_seconds": settings.api.rate_limit_window
        }
    }


@app.get("/api/v1/modules", tags=["Health"])
async def list_modules() -> Dict[str, Any]:
    """
    List all available modules with their capabilities.
    
    Returns:
        Dictionary with module information and capabilities
    """
    return {
        "modules": [
            {
                "name": "supply",
                "description": "Supply management and procurement optimization",
                "endpoint": "/api/v1/supply",
                "capabilities": [
                    "procurement_optimization",
                    "manufacturing_optimization", 
                    "supplier_management",
                    "cost_optimization"
                ],
                "input_formats": ["CSV"],
                "ai_powered": True
            },
            {
                "name": "inventory",
                "description": "Inventory management and stock optimization",
                "endpoint": "/api/v1/inventory",
                "capabilities": [
                    "safety_stock_calculation",
                    "reorder_point_optimization",
                    "stockout_prediction",
                    "inventory_tracking"
                ],
                "input_formats": ["CSV"],
                "ai_powered": True
            },
            {
                "name": "demand",
                "description": "AI-powered demand forecasting and analytics",
                "endpoint": "/api/v1/demand",
                "capabilities": [
                    "demand_forecasting",
                    "trend_analysis",
                    "seasonality_detection",
                    "pattern_recognition"
                ],
                "input_formats": ["CSV"],
                "ai_powered": True,
                "ml_model": "LightGBM"
            },
            {
                "name": "distribution",
                "description": "Distribution planning and route optimization",
                "endpoint": "/api/v1/distribution",
                "capabilities": [
                    "route_optimization",
                    "vehicle_routing",
                    "capacity_planning",
                    "cost_minimization"
                ],
                "input_formats": ["CSV"],
                "ai_powered": True,
                "algorithms": ["greedy", "nearest_neighbor"]
            }
        ],
        "total_modules": 4,
        "timestamp": time.time()
    }


# Development server configuration
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.api.debug,
        log_level="info" if not settings.api.debug else "debug",
        access_log=True
    )