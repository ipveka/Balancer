"""
Inventory Domain Data Models

This module contains Pydantic models for the inventory management domain,
including inventory status tracking, safety stock calculations, and
reorder point optimization with comprehensive validation rules.
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class TransactionType(str, Enum):
    """Enumeration for inventory transaction types."""
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"
    ADJUSTMENT = "ADJUSTMENT"
    TRANSFER = "TRANSFER"
    RETURN = "RETURN"


class RecommendedAction(str, Enum):
    """Enumeration for inventory recommendation actions."""
    REORDER = "REORDER"
    URGENT_REORDER = "URGENT_REORDER"
    SUFFICIENT_STOCK = "SUFFICIENT_STOCK"
    EXCESS_STOCK = "EXCESS_STOCK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class InventoryStatus(str, Enum):
    """Enumeration for inventory item status."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DISCONTINUED = "DISCONTINUED"
    QUARANTINE = "QUARANTINE"


# Input Models for CSV Processing

class InventoryStatusInput(BaseModel):
    """
    Input model for inventory status data processing.
    
    Represents a single SKU's current inventory status including stock levels,
    demand patterns, and service level requirements for safety stock calculation.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier", min_length=1, max_length=50)
    current_stock: int = Field(..., description="Current stock level", ge=0)
    lead_time_days: int = Field(..., description="Lead time in days", gt=0, le=365)
    service_level_target: float = Field(..., description="Target service level (0.0-1.0)", gt=0, le=1)
    avg_weekly_demand: float = Field(..., description="Average weekly demand", ge=0)
    demand_std_dev: float = Field(..., description="Standard deviation of demand", ge=0)

    @field_validator('sku')
    @classmethod
    def validate_sku_format(cls, v):
        """Validate SKU format - alphanumeric with hyphens allowed."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('SKU must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()

    @field_validator('service_level_target')
    @classmethod
    def validate_service_level_range(cls, v):
        """Validate service level is within reasonable range."""
        if v < 0.5 or v > 0.999:
            raise ValueError('Service level target must be between 0.5 (50%) and 0.999 (99.9%)')
        return v

    @model_validator(mode='after')
    def validate_demand_consistency(self):
        """Validate demand parameters are consistent."""
        if self.demand_std_dev > self.avg_weekly_demand * 2:
            raise ValueError('Demand standard deviation appears unreasonably high compared to average demand')
        
        return self


class InventoryStatusCSV(BaseModel):
    """Container model for inventory status CSV data processing."""
    data: List[InventoryStatusInput] = Field(..., description="List of inventory status records")

    @field_validator('data')
    @classmethod
    def validate_unique_skus(cls, v):
        """Ensure all SKUs in the dataset are unique."""
        skus = [item.sku for item in v]
        if len(skus) != len(set(skus)):
            raise ValueError('Duplicate SKUs found in inventory data')
        return v


# Output Models

class InventoryRecommendation(BaseModel):
    """
    Output model for inventory recommendations.
    
    Contains calculated safety stock, reorder points, and actionable
    recommendations for inventory management.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    safety_stock: int = Field(..., description="Calculated safety stock level", ge=0)
    reorder_point: int = Field(..., description="Calculated reorder point", ge=0)
    current_stock: int = Field(..., description="Current stock level", ge=0)
    recommended_action: RecommendedAction = Field(..., description="Recommended action to take")
    days_until_stockout: Optional[int] = Field(None, description="Estimated days until stockout", ge=0)
    confidence_score: float = Field(..., description="Confidence score for the recommendation", ge=0, le=1)
    calculation_details: Dict[str, Any] = Field(default_factory=dict, description="Details of safety stock calculation")

    @model_validator(mode='after')
    def validate_reorder_point_logic(self):
        """Validate reorder point is greater than or equal to safety stock."""
        if self.reorder_point < self.safety_stock:
            raise ValueError('Reorder point must be greater than or equal to safety stock')
        
        return self


class InventoryRecommendationsCSV(BaseModel):
    """Container model for inventory recommendations CSV output."""
    recommendations: List[InventoryRecommendation] = Field(..., description="List of inventory recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when recommendations were generated")
    total_recommendations: int = Field(..., description="Total number of recommendations")
    
    @model_validator(mode='after')
    def validate_count_matches(self):
        """Ensure total count matches actual recommendations."""
        if self.total_recommendations != len(self.recommendations):
            raise ValueError('Total recommendations count does not match actual recommendations')
        return self


# Internal Processing Models

class SafetyStockCalculation(BaseModel):
    """
    Internal model for safety stock calculation details.
    
    Contains intermediate calculations and parameters used in
    safety stock optimization algorithms.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    demand_variability: float = Field(..., description="Calculated demand variability", ge=0)
    lead_time_days: int = Field(..., description="Lead time in days", gt=0)
    service_level: float = Field(..., description="Target service level", gt=0, le=1)
    z_score: float = Field(..., description="Z-score for service level", gt=0)
    calculated_safety_stock: int = Field(..., description="Calculated safety stock level", ge=0)
    reorder_point: int = Field(..., description="Calculated reorder point", ge=0)
    avg_demand_during_lead_time: float = Field(..., description="Average demand during lead time", ge=0)
    demand_std_dev_during_lead_time: float = Field(..., description="Standard deviation during lead time", ge=0)

    @model_validator(mode='after')
    def validate_calculation_consistency(self):
        """Validate calculation parameters are consistent."""
        # Basic consistency check for reorder point calculation
        expected_reorder_point = self.avg_demand_during_lead_time + self.calculated_safety_stock
        if abs(self.reorder_point - expected_reorder_point) > 1:  # Allow for rounding
            raise ValueError('Reorder point calculation appears inconsistent')
        
        return self


# Core Domain Models

class InventoryItem(BaseModel):
    """
    Core inventory item model.
    
    Represents an inventory item with all relevant tracking information,
    location details, and current status.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier", min_length=1, max_length=50)
    description: str = Field(..., description="Item description", min_length=1, max_length=200)
    current_stock: int = Field(..., description="Current stock level", ge=0)
    reserved_stock: int = Field(0, description="Reserved stock level", ge=0)
    available_stock: int = Field(..., description="Available stock (current - reserved)", ge=0)
    location: str = Field(..., description="Storage location", min_length=1, max_length=50)
    unit_cost: float = Field(..., description="Unit cost", gt=0)
    reorder_point: int = Field(..., description="Reorder point", ge=0)
    safety_stock: int = Field(..., description="Safety stock level", ge=0)
    max_stock: Optional[int] = Field(None, description="Maximum stock level", ge=0)
    status: InventoryStatus = Field(InventoryStatus.ACTIVE, description="Item status")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    @field_validator('sku')
    @classmethod
    def validate_sku_format(cls, v):
        """Validate SKU format - alphanumeric with hyphens allowed."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('SKU must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()

    @model_validator(mode='after')
    def validate_stock_levels(self):
        """Validate stock level relationships."""
        # Available stock should equal current stock minus reserved stock
        if self.available_stock != self.current_stock - self.reserved_stock:
            raise ValueError('Available stock must equal current stock minus reserved stock')
        
        # Reserved stock cannot exceed current stock
        if self.reserved_stock > self.current_stock:
            raise ValueError('Reserved stock cannot exceed current stock')
        
        # Safety stock should not exceed reorder point
        if self.safety_stock > self.reorder_point:
            raise ValueError('Safety stock should not exceed reorder point')
        
        # Max stock validation if provided
        if self.max_stock and self.current_stock > self.max_stock:
            raise ValueError('Current stock cannot exceed maximum stock level')
        
        return self


class InventoryTransaction(BaseModel):
    """
    Inventory transaction model for tracking stock movements.
    
    Records all inventory movements including receipts, shipments,
    adjustments, and transfers with full audit trail.
    """
    transaction_id: str = Field(..., description="Unique transaction identifier", min_length=1, max_length=50)
    sku: str = Field(..., description="Stock Keeping Unit identifier", min_length=1, max_length=50)
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    quantity: int = Field(..., description="Transaction quantity (positive for inbound, negative for outbound)")
    unit_cost: Optional[float] = Field(None, description="Unit cost for the transaction", ge=0)
    reference_number: Optional[str] = Field(None, description="Reference number (PO, SO, etc.)", max_length=50)
    location: str = Field(..., description="Storage location", min_length=1, max_length=50)
    notes: Optional[str] = Field(None, description="Transaction notes", max_length=500)
    created_by: str = Field(..., description="User who created the transaction", min_length=1, max_length=50)
    created_at: datetime = Field(default_factory=datetime.now, description="Transaction timestamp")
    
    @field_validator('sku')
    @classmethod
    def validate_sku_format(cls, v):
        """Validate SKU format - alphanumeric with hyphens allowed."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('SKU must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()

    @field_validator('transaction_id')
    @classmethod
    def validate_transaction_id_format(cls, v):
        """Validate transaction ID format."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Transaction ID must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()

    @model_validator(mode='after')
    def validate_transaction_logic(self):
        """Validate transaction type and quantity consistency."""
        if self.transaction_type == TransactionType.INBOUND and self.quantity <= 0:
            raise ValueError('Inbound transactions must have positive quantity')
        
        if self.transaction_type == TransactionType.OUTBOUND and self.quantity >= 0:
            raise ValueError('Outbound transactions must have negative quantity')
        
        return self


# API Request/Response Models

class InventoryOptimizationRequest(BaseModel):
    """Request model for inventory optimization API endpoints."""
    csv_data: str = Field(..., description="CSV data as string")
    optimization_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional optimization parameters")


class InventoryOptimizationResponse(BaseModel):
    """Response model for inventory optimization API endpoints."""
    success: bool = Field(..., description="Whether optimization was successful")
    recommendations_count: int = Field(..., description="Number of recommendations generated")
    csv_output: str = Field(..., description="Recommendations as CSV string")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    optimization_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of optimization results")


class InventoryItemRequest(BaseModel):
    """Request model for creating/updating inventory items."""
    sku: str = Field(..., description="Stock Keeping Unit identifier", min_length=1, max_length=50)
    description: str = Field(..., description="Item description", min_length=1, max_length=200)
    current_stock: int = Field(..., description="Current stock level", ge=0)
    location: str = Field(..., description="Storage location", min_length=1, max_length=50)
    unit_cost: float = Field(..., description="Unit cost", gt=0)
    reorder_point: int = Field(..., description="Reorder point", ge=0)
    safety_stock: int = Field(..., description="Safety stock level", ge=0)
    max_stock: Optional[int] = Field(None, description="Maximum stock level", ge=0)


class InventoryTransactionRequest(BaseModel):
    """Request model for creating inventory transactions."""
    sku: str = Field(..., description="Stock Keeping Unit identifier", min_length=1, max_length=50)
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    quantity: int = Field(..., description="Transaction quantity")
    unit_cost: Optional[float] = Field(None, description="Unit cost for the transaction", ge=0)
    reference_number: Optional[str] = Field(None, description="Reference number", max_length=50)
    location: str = Field(..., description="Storage location", min_length=1, max_length=50)
    notes: Optional[str] = Field(None, description="Transaction notes", max_length=500)
    created_by: str = Field(..., description="User who created the transaction", min_length=1, max_length=50)


class InventoryQueryParams(BaseModel):
    """Query parameters for inventory filtering and aggregation."""
    sku: Optional[str] = Field(None, description="Filter by SKU")
    location: Optional[str] = Field(None, description="Filter by location")
    status: Optional[InventoryStatus] = Field(None, description="Filter by status")
    low_stock_only: Optional[bool] = Field(False, description="Show only items below reorder point")
    include_inactive: Optional[bool] = Field(False, description="Include inactive items")
    limit: Optional[int] = Field(100, description="Maximum number of results", gt=0, le=1000)
    offset: Optional[int] = Field(0, description="Number of results to skip", ge=0)


# Error Models

class InventoryException(Exception):
    """Base exception for inventory operations."""
    pass


class InsufficientInventoryException(InventoryException):
    """Exception raised when insufficient inventory is available."""
    pass


class InventoryItemNotFoundException(InventoryException):
    """Exception raised when inventory item is not found."""
    pass


class InvalidTransactionException(InventoryException):
    """Exception raised when transaction is invalid."""
    pass


class StockCalculationException(InventoryException):
    """Exception raised when stock calculation fails."""
    pass