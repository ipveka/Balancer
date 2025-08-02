"""
Supply Domain Data Models

This module contains Pydantic models for the supply management domain,
including procurement and manufacturing data models with comprehensive
validation rules and constraints.
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class SupplyMode(str, Enum):
    """Enumeration for supply operation modes."""
    PROCUREMENT = "procurement"
    MANUFACTURING = "manufacturing"


class RecommendationAction(str, Enum):
    """Enumeration for recommendation actions."""
    ORDER = "ORDER"
    PRODUCE = "PRODUCE"
    NO_ACTION = "NO_ACTION"
    URGENT_ORDER = "URGENT_ORDER"
    URGENT_PRODUCE = "URGENT_PRODUCE"


# Input Models for CSV Processing

class ProcurementDataInput(BaseModel):
    """
    Input model for procurement data processing.
    
    Represents a single SKU's procurement requirements including current
    inventory levels, demand forecasts, and supplier information.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier", min_length=1, max_length=50)
    current_inventory: int = Field(..., description="Current inventory level", ge=0)
    forecast_demand_4weeks: int = Field(..., description="Forecasted demand for next 4 weeks", ge=0)
    safety_stock: int = Field(..., description="Required safety stock level", ge=0)
    min_order_qty: int = Field(..., description="Minimum order quantity from supplier", gt=0)
    supplier_id: str = Field(..., description="Supplier identifier", min_length=1, max_length=20)
    unit_cost: float = Field(..., description="Cost per unit", gt=0)

    @field_validator('sku')
    @classmethod
    def validate_sku_format(cls, v):
        """Validate SKU format - alphanumeric with hyphens allowed."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('SKU must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()

    @field_validator('supplier_id')
    @classmethod
    def validate_supplier_id_format(cls, v):
        """Validate supplier ID format."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Supplier ID must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()


class ManufacturingDataInput(BaseModel):
    """
    Input model for manufacturing data processing.
    
    Represents a single SKU's manufacturing requirements including current
    inventory levels, demand forecasts, and production parameters.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier", min_length=1, max_length=50)
    current_inventory: int = Field(..., description="Current inventory level", ge=0)
    forecast_demand_4weeks: int = Field(..., description="Forecasted demand for next 4 weeks", ge=0)
    safety_stock: int = Field(..., description="Required safety stock level", ge=0)
    batch_size: int = Field(..., description="Manufacturing batch size", gt=0)
    production_time_days: int = Field(..., description="Production time in days", gt=0, le=365)
    unit_cost: float = Field(..., description="Production cost per unit", gt=0)

    @field_validator('sku')
    @classmethod
    def validate_sku_format(cls, v):
        """Validate SKU format - alphanumeric with hyphens allowed."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('SKU must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()

    @model_validator(mode='after')
    def validate_production_feasibility(self):
        """Validate that production parameters are feasible."""
        if self.batch_size and self.forecast_demand_4weeks and self.batch_size > self.forecast_demand_4weeks * 10:
            raise ValueError('Batch size is unreasonably large compared to demand forecast')
        
        return self


# CSV Container Models

class ProcurementDataCSV(BaseModel):
    """Container model for procurement CSV data processing."""
    data: List[ProcurementDataInput] = Field(..., description="List of procurement data records")

    @field_validator('data')
    @classmethod
    def validate_unique_skus(cls, v):
        """Ensure all SKUs in the dataset are unique."""
        skus = [item.sku for item in v]
        if len(skus) != len(set(skus)):
            raise ValueError('Duplicate SKUs found in procurement data')
        return v


class ManufacturingDataCSV(BaseModel):
    """Container model for manufacturing CSV data processing."""
    data: List[ManufacturingDataInput] = Field(..., description="List of manufacturing data records")

    @field_validator('data')
    @classmethod
    def validate_unique_skus(cls, v):
        """Ensure all SKUs in the dataset are unique."""
        skus = [item.sku for item in v]
        if len(skus) != len(set(skus)):
            raise ValueError('Duplicate SKUs found in manufacturing data')
        return v


# Output Models

class ProcurementRecommendation(BaseModel):
    """
    Output model for procurement recommendations.
    
    Contains optimized procurement recommendations including quantities,
    timing, and cost calculations.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    recommended_quantity: int = Field(..., description="Recommended order quantity", ge=0)
    supplier_id: str = Field(..., description="Recommended supplier identifier")
    order_date: str = Field(..., description="Recommended order date in YYYY-MM-DD format")
    expected_delivery: str = Field(..., description="Expected delivery date in YYYY-MM-DD format")
    total_cost: float = Field(..., description="Total cost for the recommended order", ge=0)
    recommendation_action: RecommendationAction = Field(..., description="Recommended action to take")
    confidence_score: float = Field(..., description="Confidence score for the recommendation", ge=0, le=1)

    @field_validator('order_date', 'expected_delivery')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format is YYYY-MM-DD."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v

    @model_validator(mode='after')
    def validate_delivery_after_order(self):
        """Ensure delivery date is after order date."""
        if self.order_date and self.expected_delivery:
            order_dt = datetime.strptime(self.order_date, '%Y-%m-%d')
            delivery_dt = datetime.strptime(self.expected_delivery, '%Y-%m-%d')
            
            if delivery_dt <= order_dt:
                raise ValueError('Expected delivery date must be after order date')
        
        return self


class ManufacturingRecommendation(BaseModel):
    """
    Output model for manufacturing recommendations.
    
    Contains optimized manufacturing recommendations including batch quantities,
    production scheduling, and cost calculations.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    recommended_batch_qty: int = Field(..., description="Recommended production batch quantity", ge=0)
    production_start_date: str = Field(..., description="Recommended production start date in YYYY-MM-DD format")
    production_complete_date: str = Field(..., description="Expected production completion date in YYYY-MM-DD format")
    total_cost: float = Field(..., description="Total production cost for the recommended batch", ge=0)
    recommendation_action: RecommendationAction = Field(..., description="Recommended action to take")
    confidence_score: float = Field(..., description="Confidence score for the recommendation", ge=0, le=1)

    @field_validator('production_start_date', 'production_complete_date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format is YYYY-MM-DD."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v

    @model_validator(mode='after')
    def validate_completion_after_start(self):
        """Ensure completion date is after start date."""
        if self.production_start_date and self.production_complete_date:
            start_dt = datetime.strptime(self.production_start_date, '%Y-%m-%d')
            complete_dt = datetime.strptime(self.production_complete_date, '%Y-%m-%d')
            
            if complete_dt <= start_dt:
                raise ValueError('Production completion date must be after start date')
        
        return self


# CSV Output Container Models

class ProcurementRecommendationsCSV(BaseModel):
    """Container model for procurement recommendations CSV output."""
    recommendations: List[ProcurementRecommendation] = Field(..., description="List of procurement recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when recommendations were generated")
    total_recommendations: int = Field(..., description="Total number of recommendations")
    
    @model_validator(mode='after')
    def validate_count_matches(self):
        """Ensure total count matches actual recommendations."""
        if self.total_recommendations != len(self.recommendations):
            raise ValueError('Total recommendations count does not match actual recommendations')
        return self


class ManufacturingRecommendationsCSV(BaseModel):
    """Container model for manufacturing recommendations CSV output."""
    recommendations: List[ManufacturingRecommendation] = Field(..., description="List of manufacturing recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when recommendations were generated")
    total_recommendations: int = Field(..., description="Total number of recommendations")
    
    @model_validator(mode='after')
    def validate_count_matches(self):
        """Ensure total count matches actual recommendations."""
        if self.total_recommendations != len(self.recommendations):
            raise ValueError('Total recommendations count does not match actual recommendations')
        return self


# Internal Processing Models

class SupplyOptimizationResult(BaseModel):
    """
    Internal model for supply optimization calculations.
    
    Contains intermediate calculations and optimization results used
    internally by the supply service layer.
    """
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    mode: SupplyMode = Field(..., description="Supply operation mode")
    current_inventory: int = Field(..., description="Current inventory level")
    demand_forecast: int = Field(..., description="Demand forecast")
    safety_stock_requirement: int = Field(..., description="Safety stock requirement")
    calculated_need: int = Field(..., description="Calculated supply need")
    optimized_quantity: int = Field(..., description="Optimized supply quantity")
    cost_per_unit: float = Field(..., description="Cost per unit")
    total_cost: float = Field(..., description="Total calculated cost")
    optimization_factors: Dict[str, Any] = Field(default_factory=dict, description="Factors used in optimization")
    
    @model_validator(mode='after')
    def validate_calculated_need(self):
        """Validate calculated need is reasonable."""
        expected_need = max(0, self.demand_forecast + self.safety_stock_requirement - self.current_inventory)
        if abs(self.calculated_need - expected_need) > expected_need * 0.1:  # Allow 10% variance
            raise ValueError('Calculated need appears inconsistent with input parameters')
        
        return self


# Supplier and Contact Information Models

class ContactInfo(BaseModel):
    """
    Contact information model for suppliers and manufacturers.
    
    Contains contact details and communication preferences for
    supply chain partners.
    """
    name: str = Field(..., description="Contact person name", min_length=1, max_length=100)
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    role: Optional[str] = Field(None, description="Role or title", max_length=50)
    
    @field_validator('email')
    @classmethod
    def validate_email_format(cls, v):
        """Basic email format validation."""
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v

    @field_validator('phone')
    @classmethod
    def validate_phone_format(cls, v):
        """Basic phone format validation."""
        if v:
            # Remove common separators and check if remaining chars are digits
            cleaned = v.replace('-', '').replace(' ', '').replace('(', '').replace(')', '').replace('+', '')
            if not cleaned.isdigit() or len(cleaned) < 10:
                raise ValueError('Phone number must contain at least 10 digits')
        return v


class Supplier(BaseModel):
    """
    Supplier model for procurement operations.
    
    Contains supplier information, capabilities, and performance metrics
    used in procurement decision making.
    """
    supplier_id: str = Field(..., description="Unique supplier identifier", min_length=1, max_length=20)
    name: str = Field(..., description="Supplier company name", min_length=1, max_length=200)
    contact_info: ContactInfo = Field(..., description="Primary contact information")
    lead_time_days: int = Field(..., description="Standard lead time in days", gt=0, le=365)
    minimum_order_value: float = Field(0, description="Minimum order value", ge=0)
    payment_terms: Optional[str] = Field(None, description="Payment terms", max_length=100)
    quality_rating: float = Field(..., description="Quality rating (0-5 scale)", ge=0, le=5)
    delivery_performance: float = Field(..., description="On-time delivery percentage", ge=0, le=1)
    active: bool = Field(True, description="Whether supplier is active")
    
    @field_validator('supplier_id')
    @classmethod
    def validate_supplier_id_format(cls, v):
        """Validate supplier ID format."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Supplier ID must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()


class PurchaseOrder(BaseModel):
    """
    Purchase order model for procurement tracking.
    
    Represents a purchase order with all relevant details for
    procurement management and tracking.
    """
    po_number: str = Field(..., description="Purchase order number", min_length=1, max_length=50)
    supplier_id: str = Field(..., description="Supplier identifier")
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    quantity: int = Field(..., description="Ordered quantity", gt=0)
    unit_cost: float = Field(..., description="Cost per unit", gt=0)
    total_cost: float = Field(..., description="Total order cost", gt=0)
    order_date: date = Field(..., description="Order placement date")
    expected_delivery: date = Field(..., description="Expected delivery date")
    status: str = Field(..., description="Order status")
    notes: Optional[str] = Field(None, description="Additional notes", max_length=500)
    
    @model_validator(mode='after')
    def validate_total_cost_and_delivery(self):
        """Validate total cost and delivery date."""
        # Validate total cost matches quantity * unit_cost
        expected_total = self.quantity * self.unit_cost
        if abs(self.total_cost - expected_total) > 0.01:  # Allow for small rounding differences
            raise ValueError('Total cost must equal quantity * unit_cost')
        
        # Ensure delivery date is after order date
        if self.expected_delivery <= self.order_date:
            raise ValueError('Expected delivery date must be after order date')
        
        return self


# API Request/Response Models

class SupplyOptimizationRequest(BaseModel):
    """Request model for supply optimization API endpoints."""
    mode: SupplyMode = Field(..., description="Supply operation mode")
    csv_data: str = Field(..., description="CSV data as string")
    optimization_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional optimization parameters")


class SupplyOptimizationResponse(BaseModel):
    """Response model for supply optimization API endpoints."""
    success: bool = Field(..., description="Whether optimization was successful")
    mode: SupplyMode = Field(..., description="Supply operation mode used")
    recommendations_count: int = Field(..., description="Number of recommendations generated")
    csv_output: str = Field(..., description="Recommendations as CSV string")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    optimization_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of optimization results")


# Error Models

class SupplyChainException(Exception):
    """Base exception for supply chain operations."""
    pass


class SupplierNotFoundException(SupplyChainException):
    """Exception raised when supplier is not found."""
    pass


class InvalidSupplyModeException(SupplyChainException):
    """Exception raised when invalid supply mode is specified."""
    pass


class OptimizationFailedException(SupplyChainException):
    """Exception raised when optimization fails."""
    pass