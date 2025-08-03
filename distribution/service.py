"""
Distribution Domain Service Layer

This module provides the main service interface for distribution optimization,
including vehicle routing and logistics operations.
All functions work with pandas DataFrames as the primary data format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .models import (
    OrderInput, VehicleInput, RouteAssignment,
    VRPSolution, DistanceMatrix, VehicleCapacityUtilization
)
from utils.ml_utils import (
    haversine_distance, create_distance_matrix, VRPSolver
)

logger = logging.getLogger(__name__)


class DistributionService:
    """
    Main service class for distribution optimization operations.
    
    This service handles vehicle routing optimization, route planning,
    and logistics operations using pandas DataFrames as the primary data format.
    """
    
    def __init__(self):
        """Initialize the distribution service."""
        self.logger = logging.getLogger(__name__)
    
    # =============================================================================
    # CORE ROUTE OPTIMIZATION
    # =============================================================================
    
    def optimize_routes(
        self, 
        orders_data: pd.DataFrame, 
        vehicles_data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize vehicle routes for delivery orders.
        
        Args:
            orders_data (pd.DataFrame): Orders data with required columns:
                - order_id: Unique identifier for the order
                - customer_lat: Customer latitude coordinate
                - customer_lon: Customer longitude coordinate
                - volume_m3: Order volume in cubic meters
                - weight_kg: Order weight in kilograms
            vehicles_data (pd.DataFrame): Vehicles data with required columns:
                - vehicle_id: Unique identifier for the vehicle
                - max_volume_m3: Maximum volume capacity
                - max_weight_kg: Maximum weight capacity
                - cost_per_km: Operating cost per kilometer
            params (Dict[str, Any], optional): Optimization parameters
                
        Returns:
            Dict[str, Any]: Route optimization results
        """
        # Validate input data
        orders_required = ['order_id', 'customer_lat', 'customer_lon', 'volume_m3', 'weight_kg']
        vehicles_required = ['vehicle_id', 'max_volume_m3', 'max_weight_kg', 'cost_per_km']
        
        self._validate_dataframe_columns(orders_data, orders_required)
        self._validate_dataframe_columns(vehicles_data, vehicles_required)
        
        # Set default parameters
        if params is None:
            params = {}
        
        optimization_params = {
            'algorithm_preference': params.get('algorithm_preference', 'greedy'),
            'max_route_distance_km': params.get('max_route_distance_km', 200.0),
            'max_route_duration_hours': params.get('max_route_duration_hours', 8.0),
            'optimization_timeout_seconds': params.get('optimization_timeout_seconds', 300),
            'depot_location': params.get('depot_location', {'lat': 40.7128, 'lon': -74.0060})
        }
        
        processing_start = datetime.now()
        
        try:
            # Create locations list (depot + customer locations)
            depot = optimization_params['depot_location']
            locations = [{'lat': depot['lat'], 'lon': depot['lon']}]  # Depot first
            
            # Add customer locations
            for _, order in orders_data.iterrows():
                locations.append({
                    'lat': order['customer_lat'],
                    'lon': order['customer_lon']
                })
            
            # Create distance matrix
            distance_matrix = create_distance_matrix(locations, method="haversine")
            
            # Prepare demands (excluding depot)
            demands = orders_data['weight_kg'].tolist()
            
            # Prepare vehicle capacities
            capacities = vehicles_data['max_weight_kg'].tolist()
            
            # Solve VRP
            vrp_solver = VRPSolver(
                distance_matrix=distance_matrix,
                demands=demands,
                vehicle_capacities=capacities,
                depot_index=0
            )
            
            # Choose algorithm
            if optimization_params['algorithm_preference'] == 'nearest_neighbor':
                vrp_solution = vrp_solver.nearest_neighbor_assignment()
            else:
                vrp_solution = vrp_solver.greedy_assignment()
            
            # Convert solution to structured format
            assignments = []
            routes = []
            vehicle_utilization = {}
            
            for route_info in vrp_solution['routes']:
                vehicle_idx = route_info['vehicle_id']
                vehicle_id = vehicles_data.iloc[vehicle_idx]['vehicle_id']
                cost_per_km = vehicles_data.iloc[vehicle_idx]['cost_per_km']
                
                route_orders = []
                total_volume = 0
                total_weight = 0
                
                # Process each stop in the route (excluding depot visits)
                sequence = 1
                for i, location_idx in enumerate(route_info['route'][1:-1], 1):  # Skip depot
                    order_idx = location_idx - 1  # Adjust for depot offset
                    if order_idx < len(orders_data):
                        order = orders_data.iloc[order_idx]
                        
                        # Calculate distance to this stop
                        if i == 1:
                            # Distance from depot to first customer
                            prev_location_idx = 0
                        else:
                            prev_location_idx = route_info['route'][i-1]
                        
                        distance_km = distance_matrix[prev_location_idx][location_idx]
                        total_cost = distance_km * cost_per_km
                        
                        assignment = {
                            'vehicle_id': vehicle_id,
                            'order_id': order['order_id'],
                            'sequence': sequence,
                            'distance_km': round(distance_km, 2),
                            'total_cost': round(total_cost, 2)
                        }
                        
                        assignments.append(assignment)
                        route_orders.append(order['order_id'])
                        
                        total_volume += order['volume_m3']
                        total_weight += order['weight_kg']
                        sequence += 1
                
                # Create route summary
                if route_orders:
                    route = {
                        'route_id': f"ROUTE-{vehicle_id}",
                        'vehicle_id': vehicle_id,
                        'order_sequence': route_orders,
                        'total_distance_km': round(route_info['distance'], 2),
                        'total_cost': round(route_info['distance'] * cost_per_km, 2),
                        'estimated_duration_hours': round(route_info['distance'] / 50, 2)  # Assume 50 km/h
                    }
                    routes.append(route)
                    
                    # Calculate vehicle utilization
                    vehicle_capacity = vehicles_data.iloc[vehicle_idx]
                    volume_utilization = (total_volume / vehicle_capacity['max_volume_m3']) * 100
                    weight_utilization = (total_weight / vehicle_capacity['max_weight_kg']) * 100
                    
                    vehicle_utilization[vehicle_id] = {
                        'volume_utilization_percent': round(volume_utilization, 1),
                        'weight_utilization_percent': round(weight_utilization, 1),
                        'overall_utilization_percent': round(max(volume_utilization, weight_utilization), 1)
                    }
            
            # Create solution summary
            processing_time = (datetime.now() - processing_start).total_seconds()
            
            solution = {
                'solution_id': f"SOL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'total_distance': round(vrp_solution['total_distance'], 2),
                'total_cost': round(sum(r['total_cost'] for r in routes), 2),
                'vehicle_utilization': {k: v['overall_utilization_percent'] for k, v in vehicle_utilization.items()},
                'algorithm_used': vrp_solution['algorithm'],
                'optimization_time_seconds': round(processing_time, 3),
                'num_routes': len(routes),
                'num_orders_assigned': len(assignments),
                'num_orders_unassigned': len(vrp_solution['unassigned_locations'])
            }
            
            # Create DataFrames for results
            assignments_df = pd.DataFrame(assignments)
            routes_df = pd.DataFrame(routes)
            
            return {
                'success': True,
                'solution': solution,
                'routes': routes_df,
                'assignments': assignments_df,
                'vehicle_utilization': vehicle_utilization,
                'optimization_status': 'success' if len(vrp_solution['unassigned_locations']) == 0 else 'partial',
                'unassigned_orders': vrp_solution['unassigned_locations'],
                'processing_time_seconds': round(processing_time, 3)
            }
            
        except Exception as e:
            self.logger.error(f"Route optimization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'optimization_status': 'failed'
            }
    
    def calculate_distance_matrix(
        self, 
        locations_data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate distance matrix for locations.
        
        Args:
            locations_data (pd.DataFrame): Locations data with columns:
                - location_id: Unique identifier for the location
                - latitude: Latitude coordinate
                - longitude: Longitude coordinate
            params (Dict[str, Any], optional): Calculation parameters
                
        Returns:
            Dict[str, Any]: Distance matrix results
        """
        required_columns = ['location_id', 'latitude', 'longitude']
        self._validate_dataframe_columns(locations_data, required_columns)
        
        if params is None:
            params = {}
        
        calculation_method = params.get('calculation_method', 'haversine')
        
        # Prepare locations for distance calculation
        locations = []
        for _, row in locations_data.iterrows():
            locations.append({
                'lat': row['latitude'],
                'lon': row['longitude']
            })
        
        # Calculate distance matrix
        distance_matrix = create_distance_matrix(locations, method=calculation_method)
        
        # Convert to dictionary format with location IDs
        location_ids = locations_data['location_id'].tolist()
        distances_dict = {}
        
        for i, from_id in enumerate(location_ids):
            distances_dict[from_id] = {}
            for j, to_id in enumerate(location_ids):
                distances_dict[from_id][to_id] = round(distance_matrix[i][j], 2)
        
        return {
            'success': True,
            'distance_matrix': distances_dict,
            'calculation_method': calculation_method,
            'locations_count': len(location_ids),
            'matrix_id': f"MATRIX-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    
    def analyze_route_performance(
        self, 
        routes_data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze route performance metrics.
        
        Args:
            routes_data (pd.DataFrame): Route data with columns:
                - vehicle_id: Vehicle identifier
                - total_distance_km: Total route distance
                - total_cost: Total route cost
                - estimated_duration_hours: Estimated duration
            params (Dict[str, Any], optional): Analysis parameters
                
        Returns:
            Dict[str, Any]: Performance analysis results
        """
        required_columns = ['vehicle_id', 'total_distance_km', 'total_cost']
        self._validate_dataframe_columns(routes_data, required_columns)
        
        # Calculate performance metrics
        performance_metrics = {
            'total_routes': len(routes_data),
            'total_distance_km': routes_data['total_distance_km'].sum(),
            'total_cost': routes_data['total_cost'].sum(),
            'avg_route_distance': routes_data['total_distance_km'].mean(),
            'avg_route_cost': routes_data['total_cost'].mean(),
            'cost_per_km': routes_data['total_cost'].sum() / routes_data['total_distance_km'].sum() if routes_data['total_distance_km'].sum() > 0 else 0,
            'longest_route_distance': routes_data['total_distance_km'].max(),
            'shortest_route_distance': routes_data['total_distance_km'].min(),
            'most_expensive_route': routes_data['total_cost'].max(),
            'least_expensive_route': routes_data['total_cost'].min()
        }
        
        # Add duration metrics if available
        if 'estimated_duration_hours' in routes_data.columns:
            performance_metrics.update({
                'total_duration_hours': routes_data['estimated_duration_hours'].sum(),
                'avg_duration_hours': routes_data['estimated_duration_hours'].mean(),
                'longest_duration_hours': routes_data['estimated_duration_hours'].max(),
                'shortest_duration_hours': routes_data['estimated_duration_hours'].min()
            })
        
        # Calculate efficiency metrics
        efficiency_metrics = []
        for _, route in routes_data.iterrows():
            efficiency = {
                'vehicle_id': route['vehicle_id'],
                'distance_km': route['total_distance_km'],
                'cost': route['total_cost'],
                'cost_per_km': route['total_cost'] / route['total_distance_km'] if route['total_distance_km'] > 0 else 0,
                'efficiency_score': self._calculate_route_efficiency_score(route)
            }
            efficiency_metrics.append(efficiency)
        
        efficiency_df = pd.DataFrame(efficiency_metrics)
        
        return {
            'success': True,
            'performance_metrics': performance_metrics,
            'route_efficiency': efficiency_df,
            'recommendations': self._generate_performance_recommendations(performance_metrics, efficiency_df)
        }
    
    def analyze_capacity_utilization(
        self, 
        assignments_data: pd.DataFrame, 
        vehicles_data: pd.DataFrame, 
        orders_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze vehicle capacity utilization.
        
        Args:
            assignments_data (pd.DataFrame): Route assignments
            vehicles_data (pd.DataFrame): Vehicle specifications
            orders_data (pd.DataFrame): Order details with volume/weight
                
        Returns:
            Dict[str, Any]: Capacity utilization analysis
        """
        # Merge data to get complete picture
        merged_data = assignments_data.merge(
            orders_data[['order_id', 'volume_m3', 'weight_kg']], 
            on='order_id'
        )
        
        # Calculate utilization by vehicle
        utilization_results = []
        
        for vehicle_id in merged_data['vehicle_id'].unique():
            vehicle_assignments = merged_data[merged_data['vehicle_id'] == vehicle_id]
            vehicle_spec = vehicles_data[vehicles_data['vehicle_id'] == vehicle_id].iloc[0]
            
            total_volume_used = vehicle_assignments['volume_m3'].sum()
            total_weight_used = vehicle_assignments['weight_kg'].sum()
            
            volume_utilization = (total_volume_used / vehicle_spec['max_volume_m3']) * 100
            weight_utilization = (total_weight_used / vehicle_spec['max_weight_kg']) * 100
            
            is_overloaded = (total_volume_used > vehicle_spec['max_volume_m3'] or 
                           total_weight_used > vehicle_spec['max_weight_kg'])
            
            utilization = {
                'vehicle_id': vehicle_id,
                'total_volume_used_m3': round(total_volume_used, 2),
                'total_weight_used_kg': round(total_weight_used, 2),
                'max_volume_m3': vehicle_spec['max_volume_m3'],
                'max_weight_kg': vehicle_spec['max_weight_kg'],
                'volume_utilization_percent': round(volume_utilization, 1),
                'weight_utilization_percent': round(weight_utilization, 1),
                'overall_utilization_percent': round(max(volume_utilization, weight_utilization), 1),
                'is_overloaded': is_overloaded,
                'orders_assigned': len(vehicle_assignments)
            }
            
            utilization_results.append(utilization)
        
        utilization_df = pd.DataFrame(utilization_results)
        
        # Calculate summary statistics
        summary_stats = {
            'avg_volume_utilization': utilization_df['volume_utilization_percent'].mean(),
            'avg_weight_utilization': utilization_df['weight_utilization_percent'].mean(),
            'avg_overall_utilization': utilization_df['overall_utilization_percent'].mean(),
            'overloaded_vehicles': len(utilization_df[utilization_df['is_overloaded']]),
            'underutilized_vehicles': len(utilization_df[utilization_df['overall_utilization_percent'] < 70]),
            'well_utilized_vehicles': len(utilization_df[
                (utilization_df['overall_utilization_percent'] >= 70) & 
                (utilization_df['overall_utilization_percent'] <= 95)
            ])
        }
        
        return {
            'success': True,
            'vehicle_utilization': utilization_df,
            'summary_statistics': summary_stats,
            'recommendations': self._generate_utilization_recommendations(utilization_df, summary_stats)
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
    
    def _calculate_route_efficiency_score(self, route: pd.Series) -> float:
        """Calculate efficiency score for a route (0-100)."""
        # Simple efficiency calculation based on cost per km
        cost_per_km = route['total_cost'] / route['total_distance_km'] if route['total_distance_km'] > 0 else 0
        
        # Normalize to 0-100 scale (assuming $2.00/km is baseline)
        baseline_cost_per_km = 2.0
        if cost_per_km <= baseline_cost_per_km:
            return 100.0
        else:
            return max(0, 100 - ((cost_per_km - baseline_cost_per_km) / baseline_cost_per_km) * 50)
    
    def _generate_performance_recommendations(
        self, 
        performance_metrics: Dict[str, float], 
        efficiency_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # High cost per km recommendation
        if performance_metrics['cost_per_km'] > 2.0:
            recommendations.append({
                'category': 'COST_OPTIMIZATION',
                'priority': 'HIGH',
                'recommendation': 'Consider route consolidation or vehicle optimization to reduce cost per kilometer',
                'current_cost_per_km': performance_metrics['cost_per_km'],
                'target_cost_per_km': 2.0
            })
        
        # Route length variation recommendation
        distance_std = efficiency_df['distance_km'].std()
        if distance_std > performance_metrics['avg_route_distance'] * 0.5:
            recommendations.append({
                'category': 'ROUTE_BALANCING',
                'priority': 'MEDIUM',
                'recommendation': 'Routes have high variation in distance. Consider rebalancing for more consistent workloads',
                'distance_variation': distance_std
            })
        
        # Low efficiency routes
        low_efficiency_routes = efficiency_df[efficiency_df['efficiency_score'] < 70]
        if not low_efficiency_routes.empty:
            recommendations.append({
                'category': 'EFFICIENCY_IMPROVEMENT',
                'priority': 'MEDIUM',
                'recommendation': f'{len(low_efficiency_routes)} routes have low efficiency scores. Review routing algorithms',
                'affected_vehicles': low_efficiency_routes['vehicle_id'].tolist()
            })
        
        return recommendations
    
    def _generate_utilization_recommendations(
        self, 
        utilization_df: pd.DataFrame, 
        summary_stats: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate capacity utilization recommendations."""
        recommendations = []
        
        # Overloaded vehicles
        if summary_stats['overloaded_vehicles'] > 0:
            overloaded = utilization_df[utilization_df['is_overloaded']]
            recommendations.append({
                'category': 'CAPACITY_VIOLATION',
                'priority': 'HIGH',
                'recommendation': 'Some vehicles are overloaded. Redistribute orders or use larger vehicles',
                'affected_vehicles': overloaded['vehicle_id'].tolist(),
                'count': len(overloaded)
            })
        
        # Underutilized vehicles
        if summary_stats['underutilized_vehicles'] > 0:
            underutilized = utilization_df[utilization_df['overall_utilization_percent'] < 70]
            recommendations.append({
                'category': 'UNDERUTILIZATION',
                'priority': 'MEDIUM',
                'recommendation': 'Some vehicles are underutilized. Consider consolidating routes or using smaller vehicles',
                'affected_vehicles': underutilized['vehicle_id'].tolist(),
                'avg_utilization': underutilized['overall_utilization_percent'].mean()
            })
        
        # Overall utilization
        if summary_stats['avg_overall_utilization'] < 75:
            recommendations.append({
                'category': 'FLEET_OPTIMIZATION',
                'priority': 'LOW',
                'recommendation': 'Overall fleet utilization is below optimal. Consider fleet size optimization',
                'current_utilization': summary_stats['avg_overall_utilization'],
                'target_utilization': 80.0
            })
        
        return recommendations
    
    # =============================================================================
    # DATA FORMAT CONVERSION UTILITIES
    # =============================================================================
    
    def from_csv(self, csv_data: str, data_type: str = 'orders') -> pd.DataFrame:
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