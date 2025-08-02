"""
Distribution domain data models and validation schemas.

This module contains Pydantic models for distribution management, including
distribution centers, delivery routes, vehicle routing problem (VRP) solutions,
and logistics optimization. All models include comprehensive validation for
geographical data, capacity constraints, and route optimization parameters.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
import math


class OrderInput(BaseModel):
    """
    Input model for customer orders in the distribution system.
    
    Represents a single customer order with location and capacity requirements
    for route optimization and vehicle assignment.
    """
    order_id: str = Field(..., description="Unique identifier for the order")
    customer_lat: float = Field(..., ge=-90, le=90, description="Customer latitude coordinate")
    customer_lon: float = Field(..., ge=-180, le=180, description="Customer longitude coordinate")
    volume_m3: float = Field(..., gt=0, description="Order volume in cubic meters")
    weight_kg: float = Field(..., gt=0, description="Order weight in kilograms")
    
    @field_validator('customer_lat')
    @classmethod
    def validate_latitude(cls, v):
        """Validate latitude is within valid range."""
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90 degrees')
        return v
    
    @field_validator('customer_lon')
    @classmethod
    def validate_longitude(cls, v):
        """Validate longitude is within valid range."""
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180 degrees')
        return v


class VehicleInput(BaseModel):
    """
    Input model for delivery vehicles in the distribution system.
    
    Represents a delivery vehicle with capacity constraints and cost parameters
    for route optimization and assignment algorithms.
    """
    vehicle_id: str = Field(..., description="Unique identifier for the vehicle")
    max_volume_m3: float = Field(..., gt=0, description="Maximum volume capacity in cubic meters")
    max_weight_kg: float = Field(..., gt=0, description="Maximum weight capacity in kilograms")
    cost_per_km: float = Field(..., gt=0, description="Operating cost per kilometer")
    
    @field_validator('max_volume_m3', 'max_weight_kg', 'cost_per_km')
    @classmethod
    def validate_positive_values(cls, v):
        """Ensure all capacity and cost values are positive."""
        if v <= 0:
            raise ValueError('Value must be greater than zero')
        return v


class OrdersAndVehiclesCSV(BaseModel):
    """
    CSV input model containing both orders and vehicles data.
    
    This model represents the complete input dataset for distribution optimization,
    combining customer orders and available vehicles for route planning.
    """
    orders: List[OrderInput] = Field(..., description="List of customer orders")
    vehicles: List[VehicleInput] = Field(..., description="List of available vehicles")
    
    @field_validator('orders')
    @classmethod
    def validate_orders_not_empty(cls, v):
        """Ensure at least one order is provided."""
        if not v:
            raise ValueError('At least one order must be provided')
        return v
    
    @field_validator('vehicles')
    @classmethod
    def validate_vehicles_not_empty(cls, v):
        """Ensure at least one vehicle is provided."""
        if not v:
            raise ValueError('At least one vehicle must be provided')
        return v


class RouteAssignment(BaseModel):
    """
    Output model for route assignments after optimization.
    
    Represents the assignment of a specific order to a vehicle with routing
    information including sequence, distance, and cost calculations.
    """
    vehicle_id: str = Field(..., description="ID of the assigned vehicle")
    order_id: str = Field(..., description="ID of the assigned order")
    sequence: int = Field(..., ge=1, description="Order of delivery in the route (1-based)")
    distance_km: float = Field(..., ge=0, description="Distance to this order in kilometers")
    total_cost: float = Field(..., ge=0, description="Total cost for this assignment")
    
    @field_validator('sequence')
    @classmethod
    def validate_sequence_positive(cls, v):
        """Ensure sequence number is positive."""
        if v < 1:
            raise ValueError('Sequence must be 1 or greater')
        return v


class RouteAssignmentsCSV(BaseModel):
    """
    CSV output model containing all route assignments.
    
    This model represents the complete output of the distribution optimization
    process, containing all order-to-vehicle assignments with routing details.
    """
    assignments: List[RouteAssignment] = Field(..., description="List of route assignments")


class DistributionCenter(BaseModel):
    """
    Model representing a distribution center or depot.
    
    Contains location information, capacity constraints, and operational
    parameters for distribution centers in the logistics network.
    """
    center_id: str = Field(..., description="Unique identifier for the distribution center")
    name: str = Field(..., description="Human-readable name of the center")
    latitude: float = Field(..., ge=-90, le=90, description="Distribution center latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Distribution center longitude")
    max_vehicles: int = Field(..., gt=0, description="Maximum number of vehicles that can operate from this center")
    operating_hours_start: str = Field(..., description="Operating hours start time (HH:MM format)")
    operating_hours_end: str = Field(..., description="Operating hours end time (HH:MM format)")
    is_active: bool = Field(default=True, description="Whether the center is currently active")
    
    @field_validator('operating_hours_start', 'operating_hours_end')
    @classmethod
    def validate_time_format(cls, v):
        """Validate time format is HH:MM."""
        try:
            hours, minutes = v.split(':')
            if not (0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError('Time must be in HH:MM format with valid hours (0-23) and minutes (0-59)')
        return v


class DeliveryRoute(BaseModel):
    """
    Model representing a complete delivery route.
    
    Contains the sequence of orders, vehicle assignment, and route metrics
    for a complete delivery route from a distribution center.
    """
    route_id: str = Field(..., description="Unique identifier for the route")
    vehicle_id: str = Field(..., description="ID of the assigned vehicle")
    distribution_center_id: str = Field(..., description="ID of the originating distribution center")
    order_sequence: List[str] = Field(..., description="Ordered list of order IDs in delivery sequence")
    total_distance_km: float = Field(..., ge=0, description="Total route distance in kilometers")
    total_cost: float = Field(..., ge=0, description="Total route cost")
    estimated_duration_hours: float = Field(..., gt=0, description="Estimated route duration in hours")
    route_date: datetime = Field(..., description="Planned route execution date")
    status: str = Field(default="planned", description="Route status (planned, in_progress, completed, cancelled)")
    
    @field_validator('order_sequence')
    @classmethod
    def validate_order_sequence_not_empty(cls, v):
        """Ensure route has at least one order."""
        if not v:
            raise ValueError('Route must contain at least one order')
        return v
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate route status is one of allowed values."""
        allowed_statuses = ["planned", "in_progress", "completed", "cancelled"]
        if v not in allowed_statuses:
            raise ValueError(f'Status must be one of: {", ".join(allowed_statuses)}')
        return v


class VRPSolution(BaseModel):
    """
    Model representing a complete Vehicle Routing Problem (VRP) solution.
    
    Contains optimization results, performance metrics, and algorithm
    information for the complete route optimization solution.
    """
    solution_id: str = Field(..., description="Unique identifier for the VRP solution")
    total_distance: float = Field(..., ge=0, description="Total distance across all routes")
    total_cost: float = Field(..., ge=0, description="Total cost across all routes")
    vehicle_utilization: Dict[str, float] = Field(..., description="Vehicle utilization percentages by vehicle ID")
    algorithm_used: str = Field(..., description="Algorithm used for optimization")
    optimization_time_seconds: float = Field(..., gt=0, description="Time taken for optimization in seconds")
    num_routes: int = Field(..., ge=0, description="Number of routes in the solution")
    num_orders_assigned: int = Field(..., ge=0, description="Number of orders successfully assigned")
    num_orders_unassigned: int = Field(..., ge=0, description="Number of orders that could not be assigned")
    created_at: datetime = Field(default_factory=datetime.now, description="Solution creation timestamp")
    
    @field_validator('algorithm_used')
    @classmethod
    def validate_algorithm(cls, v):
        """Validate algorithm is one of supported types."""
        allowed_algorithms = ["greedy", "nearest_neighbor", "genetic", "simulated_annealing"]
        if v not in allowed_algorithms:
            raise ValueError(f'Algorithm must be one of: {", ".join(allowed_algorithms)}')
        return v
    
    @field_validator('vehicle_utilization')
    @classmethod
    def validate_utilization_percentages(cls, v):
        """Validate utilization percentages are between 0 and 100."""
        for vehicle_id, utilization in v.items():
            if not 0 <= utilization <= 100:
                raise ValueError(f'Vehicle utilization for {vehicle_id} must be between 0 and 100 percent')
        return v


class DistanceMatrix(BaseModel):
    """
    Model representing distance calculations between locations.
    
    Contains distance matrix data and calculation metadata for route
    optimization algorithms and distance-based calculations.
    """
    matrix_id: str = Field(..., description="Unique identifier for the distance matrix")
    distances: Dict[str, Dict[str, float]] = Field(..., description="Distance matrix: location_id -> location_id -> distance_km")
    calculation_method: str = Field(..., description="Method used for distance calculation")
    created_at: datetime = Field(default_factory=datetime.now, description="Matrix creation timestamp")
    
    @field_validator('calculation_method')
    @classmethod
    def validate_calculation_method(cls, v):
        """Validate calculation method is supported."""
        allowed_methods = ["euclidean", "haversine", "manhattan", "road_network"]
        if v not in allowed_methods:
            raise ValueError(f'Calculation method must be one of: {", ".join(allowed_methods)}')
        return v
    
    @field_validator('distances')
    @classmethod
    def validate_distance_matrix(cls, v):
        """Validate distance matrix structure and values."""
        for from_location, to_distances in v.items():
            for to_location, distance in to_distances.items():
                if distance < 0:
                    raise ValueError(f'Distance from {from_location} to {to_location} cannot be negative')
                # Self-distance should be zero
                if from_location == to_location and distance != 0:
                    raise ValueError(f'Distance from {from_location} to itself must be zero')
        return v


class RouteOptimizationRequest(BaseModel):
    """
    Request model for route optimization operations.
    
    Contains parameters and constraints for route optimization requests,
    including algorithm selection and optimization preferences.
    """
    request_id: str = Field(..., description="Unique identifier for the optimization request")
    orders: List[OrderInput] = Field(..., description="Orders to be optimized")
    vehicles: List[VehicleInput] = Field(..., description="Available vehicles")
    distribution_center: Optional[DistributionCenter] = Field(None, description="Distribution center (depot) information")
    algorithm_preference: str = Field(default="greedy", description="Preferred optimization algorithm")
    max_route_distance_km: Optional[float] = Field(None, gt=0, description="Maximum allowed route distance")
    max_route_duration_hours: Optional[float] = Field(None, gt=0, description="Maximum allowed route duration")
    optimization_timeout_seconds: int = Field(default=300, gt=0, description="Maximum time allowed for optimization")
    
    @field_validator('algorithm_preference')
    @classmethod
    def validate_algorithm_preference(cls, v):
        """Validate algorithm preference is supported."""
        allowed_algorithms = ["greedy", "nearest_neighbor", "genetic", "simulated_annealing"]
        if v not in allowed_algorithms:
            raise ValueError(f'Algorithm preference must be one of: {", ".join(allowed_algorithms)}')
        return v


class RouteOptimizationResponse(BaseModel):
    """
    Response model for route optimization operations.
    
    Contains the optimization results, solution quality metrics,
    and execution information for route optimization requests.
    """
    request_id: str = Field(..., description="ID of the original optimization request")
    solution: VRPSolution = Field(..., description="Complete VRP solution")
    routes: List[DeliveryRoute] = Field(..., description="Individual delivery routes")
    assignments: List[RouteAssignment] = Field(..., description="Order-to-vehicle assignments")
    optimization_status: str = Field(..., description="Status of the optimization process")
    error_message: Optional[str] = Field(None, description="Error message if optimization failed")
    
    @field_validator('optimization_status')
    @classmethod
    def validate_optimization_status(cls, v):
        """Validate optimization status is one of allowed values."""
        allowed_statuses = ["success", "partial", "failed", "timeout"]
        if v not in allowed_statuses:
            raise ValueError(f'Optimization status must be one of: {", ".join(allowed_statuses)}')
        return v


class VehicleCapacityUtilization(BaseModel):
    """
    Model for tracking vehicle capacity utilization.
    
    Contains detailed capacity usage information for vehicles
    including volume and weight utilization metrics.
    """
    vehicle_id: str = Field(..., description="ID of the vehicle")
    total_volume_used_m3: float = Field(..., ge=0, description="Total volume used in cubic meters")
    total_weight_used_kg: float = Field(..., ge=0, description="Total weight used in kilograms")
    max_volume_m3: float = Field(..., gt=0, description="Maximum volume capacity")
    max_weight_kg: float = Field(..., gt=0, description="Maximum weight capacity")
    volume_utilization_percent: float = Field(..., ge=0, le=100, description="Volume utilization percentage")
    weight_utilization_percent: float = Field(..., ge=0, le=100, description="Weight utilization percentage")
    is_overloaded: bool = Field(default=False, description="Whether vehicle is over capacity")
    
    @field_validator('volume_utilization_percent', 'weight_utilization_percent')
    @classmethod
    def validate_utilization_range(cls, v):
        """Validate utilization percentages are within valid range."""
        if not 0 <= v <= 100:
            raise ValueError('Utilization percentage must be between 0 and 100')
        return v


class GeographicalLocation(BaseModel):
    """
    Base model for geographical locations with validation.
    
    Provides common geographical coordinate validation and
    utility methods for location-based calculations.
    """
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    address: Optional[str] = Field(None, description="Human-readable address")
    
    def distance_to(self, other: 'GeographicalLocation', method: str = "haversine") -> float:
        """
        Calculate distance to another geographical location.
        
        Args:
            other: Another geographical location
            method: Calculation method ("haversine" or "euclidean")
            
        Returns:
            Distance in kilometers
        """
        if method == "haversine":
            return self._haversine_distance(other)
        elif method == "euclidean":
            return self._euclidean_distance(other)
        else:
            raise ValueError("Method must be 'haversine' or 'euclidean'")
    
    def _haversine_distance(self, other: 'GeographicalLocation') -> float:
        """Calculate haversine distance between two points."""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _euclidean_distance(self, other: 'GeographicalLocation') -> float:
        """Calculate euclidean distance between two points (approximate)."""
        # Approximate conversion: 1 degree ≈ 111 km
        lat_diff = (other.latitude - self.latitude) * 111
        lon_diff = (other.longitude - self.longitude) * 111 * math.cos(math.radians(self.latitude))
        
        return math.sqrt(lat_diff ** 2 + lon_diff ** 2)