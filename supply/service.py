"""
Supply Domain Service Layer

This module provides the main service interface for supply chain optimization,
including procurement and manufacturing planning. All functions work with
pandas DataFrames as the primary data format.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging

from .models import (
    ProcurementDataInput, ManufacturingDataInput,
    ProcurementRecommendation, ManufacturingRecommendation,
    SupplyOptimizationResult, SupplyMode, RecommendationAction
)
from utils.helpers import validate_pydantic_data, validate_data_quality
from utils.ml_utils import optimize_inventory_levels, optimize_production_batch

logger = logging.getLogger(__name__)


class SupplyService:
    """
    Main service class for supply chain optimization operations.
    
    This service handles procurement and manufacturing optimization using
    pandas DataFrames as the primary data format. It supports various
    input/output formats through helper methods.
    """
    
    def __init__(self):
        """Initialize the supply service."""
        self.logger = logging.getLogger(__name__)
    
    # =============================================================================
    # CORE PROCUREMENT OPTIMIZATION
    # =============================================================================
    
    def optimize_procurement(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize procurement orders based on demand forecasts and inventory levels.
        
        Args:
            data (pd.DataFrame): Procurement data with required columns:
                - sku: Stock Keeping Unit identifier
                - current_inventory: Current inventory level
                - forecast_demand_4weeks: Forecasted demand for next 4 weeks
                - safety_stock: Required safety stock level
                - min_order_qty: Minimum order quantity from supplier
                - supplier_id: Supplier identifier
                - unit_cost: Cost per unit
            params (Dict[str, Any], optional): Optimization parameters
                
        Returns:
            Dict[str, Any]: Optimization results with recommendations DataFrame
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Validate input data
        required_columns = [
            'sku', 'current_inventory', 'forecast_demand_4weeks', 
            'safety_stock', 'min_order_qty', 'supplier_id', 'unit_cost'
        ]
        self._validate_dataframe_columns(data, required_columns)
        
        # Set default parameters
        if params is None:
            params = {}
        
        optimization_params = {
            'cost_weight': params.get('cost_weight', 0.7),
            'service_level': params.get('service_level', 0.95),
            'lead_time_buffer': params.get('lead_time_buffer', 1.2),
            'demand_variability': params.get('demand_variability', 0.15)
        }
        
        # Process each SKU
        recommendations = []
        processing_start = datetime.now()
        
        for _, row in data.iterrows():
            try:
                # Calculate procurement need
                procurement_need = max(0, 
                    row['forecast_demand_4weeks'] + 
                    row['safety_stock'] - 
                    row['current_inventory']
                )
                
                # Apply minimum order quantity constraint
                if procurement_need > 0:
                    recommended_quantity = max(procurement_need, row['min_order_qty'])
                    
                    # Apply optimization adjustments
                    recommended_quantity = self._apply_procurement_optimization(
                        recommended_quantity, row, optimization_params
                    )
                    
                    # Determine recommendation action
                    urgency_ratio = procurement_need / row['forecast_demand_4weeks']
                    if urgency_ratio > 0.8:
                        action = RecommendationAction.URGENT_ORDER
                    elif urgency_ratio > 0.3:
                        action = RecommendationAction.ORDER
                    else:
                        action = RecommendationAction.NO_ACTION
                        recommended_quantity = 0
                    
                    # Calculate dates and costs
                    order_date = datetime.now().date()
                    expected_delivery = order_date + timedelta(days=7)  # Default lead time
                    total_cost = recommended_quantity * row['unit_cost']
                    
                    # Calculate confidence score
                    confidence_score = self._calculate_confidence_score(row, optimization_params)
                    
                    recommendation = {
                        'sku': row['sku'],
                        'recommended_quantity': int(recommended_quantity),
                        'supplier_id': row['supplier_id'],
                        'order_date': order_date.strftime('%Y-%m-%d'),
                        'expected_delivery': expected_delivery.strftime('%Y-%m-%d'),
                        'total_cost': round(total_cost, 2),
                        'recommendation_action': action.value,
                        'confidence_score': round(confidence_score, 3)
                    }
                    
                    recommendations.append(recommendation)
                    
            except Exception as e:
                self.logger.error(f"Error processing SKU {row['sku']}: {str(e)}")
                continue
        
        # Create results DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        # Calculate summary metrics
        processing_time = (datetime.now() - processing_start).total_seconds()
        total_cost = recommendations_df['total_cost'].sum() if not recommendations_df.empty else 0
        
        return {
            'success': True,
            'mode': 'procurement',
            'recommendations': recommendations_df,
            'recommendations_count': len(recommendations),
            'processing_time_seconds': round(processing_time, 3),
            'optimization_summary': {
                'total_cost': total_cost,
                'total_items': len(recommendations),
                'avg_service_level': optimization_params['service_level'],
                'urgent_orders': len([r for r in recommendations if r['recommendation_action'] == 'URGENT_ORDER']),
                'regular_orders': len([r for r in recommendations if r['recommendation_action'] == 'ORDER'])
            }
        }
    
    def optimize_manufacturing(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize manufacturing schedules and batch sizes.
        
        Args:
            data (pd.DataFrame): Manufacturing data with required columns:
                - sku: Stock Keeping Unit identifier
                - current_inventory: Current inventory level
                - forecast_demand_4weeks: Forecasted demand for next 4 weeks
                - safety_stock: Required safety stock level
                - batch_size: Manufacturing batch size
                - production_time_days: Production time in days
                - unit_cost: Production cost per unit
            params (Dict[str, Any], optional): Optimization parameters
                
        Returns:
            Dict[str, Any]: Optimization results with recommendations DataFrame
        """
        # Validate input data
        required_columns = [
            'sku', 'current_inventory', 'forecast_demand_4weeks', 
            'safety_stock', 'batch_size', 'production_time_days', 'unit_cost'
        ]
        self._validate_dataframe_columns(data, required_columns)
        
        # Set default parameters
        if params is None:
            params = {}
        
        optimization_params = {
            'capacity_utilization': params.get('capacity_utilization', 0.85),
            'setup_cost_weight': params.get('setup_cost_weight', 0.3),
            'service_level': params.get('service_level', 0.95)
        }
        
        # Process each SKU
        recommendations = []
        processing_start = datetime.now()
        
        for _, row in data.iterrows():
            try:
                # Calculate production need
                production_need = max(0, 
                    row['forecast_demand_4weeks'] + 
                    row['safety_stock'] - 
                    row['current_inventory']
                )
                
                if production_need > 0:
                    # Optimize batch quantity using ML utils
                    demand_forecast = [row['forecast_demand_4weeks'] / 4] * 4  # Weekly breakdown
                    batch_optimization = optimize_production_batch(
                        demand_forecast=demand_forecast,
                        setup_cost=1000.0,  # Default setup cost
                        holding_cost_rate=0.25,
                        production_rate=row['batch_size'] / row['production_time_days']
                    )
                    
                    recommended_batch_qty = max(
                        production_need, 
                        batch_optimization['optimal_batch_size']
                    )
                    
                    # Calculate production schedule
                    production_start_date = datetime.now().date() + timedelta(days=1)
                    production_complete_date = production_start_date + timedelta(
                        days=row['production_time_days']
                    )
                    
                    # Calculate costs
                    total_cost = recommended_batch_qty * row['unit_cost']
                    
                    # Determine action
                    urgency_ratio = production_need / row['forecast_demand_4weeks']
                    if urgency_ratio > 0.8:
                        action = RecommendationAction.URGENT_PRODUCE
                    elif urgency_ratio > 0.3:
                        action = RecommendationAction.PRODUCE
                    else:
                        action = RecommendationAction.NO_ACTION
                        recommended_batch_qty = 0
                        total_cost = 0
                    
                    # Calculate confidence score
                    confidence_score = self._calculate_confidence_score(row, optimization_params)
                    
                    if action != RecommendationAction.NO_ACTION:
                        recommendation = {
                            'sku': row['sku'],
                            'recommended_batch_qty': int(recommended_batch_qty),
                            'production_start_date': production_start_date.strftime('%Y-%m-%d'),
                            'production_complete_date': production_complete_date.strftime('%Y-%m-%d'),
                            'total_cost': round(total_cost, 2),
                            'recommendation_action': action.value,
                            'confidence_score': round(confidence_score, 3)
                        }
                        
                        recommendations.append(recommendation)
                        
            except Exception as e:
                self.logger.error(f"Error processing SKU {row['sku']}: {str(e)}")
                continue
        
        # Create results DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        # Calculate summary metrics
        processing_time = (datetime.now() - processing_start).total_seconds()
        total_cost = recommendations_df['total_cost'].sum() if not recommendations_df.empty else 0
        
        return {
            'success': True,
            'mode': 'manufacturing',
            'recommendations': recommendations_df,
            'recommendations_count': len(recommendations),
            'processing_time_seconds': round(processing_time, 3),
            'optimization_summary': {
                'total_cost': total_cost,
                'total_items': len(recommendations),
                'avg_capacity_utilization': optimization_params['capacity_utilization'],
                'urgent_production': len([r for r in recommendations if r['recommendation_action'] == 'URGENT_PRODUCE']),
                'regular_production': len([r for r in recommendations if r['recommendation_action'] == 'PRODUCE'])
            }
        }
    
    # =============================================================================
    # SUPPLIER ANALYSIS
    # =============================================================================
    
    def compare_suppliers(
        self, 
        data: pd.DataFrame, 
        criteria: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple suppliers for procurement decisions.
        
        Args:
            data (pd.DataFrame): Supplier comparison data with columns:
                - sku: Stock Keeping Unit identifier
                - supplier_id: Supplier identifier
                - unit_cost: Cost per unit
                - lead_time_days: Lead time in days
                - quality_rating: Quality rating (0-5)
                - min_order_qty: Minimum order quantity
            criteria (Dict[str, float], optional): Comparison criteria weights
                
        Returns:
            Dict[str, Any]: Supplier comparison results
        """
        required_columns = ['sku', 'supplier_id', 'unit_cost', 'lead_time_days', 'quality_rating']
        self._validate_dataframe_columns(data, required_columns)
        
        if criteria is None:
            criteria = {
                'cost_weight': 0.4,
                'quality_weight': 0.3,
                'lead_time_weight': 0.3
            }
        
        # Normalize weights
        total_weight = sum(criteria.values())
        criteria = {k: v/total_weight for k, v in criteria.items()}
        
        # Group by SKU and compare suppliers
        comparison_results = []
        
        for sku, sku_data in data.groupby('sku'):
            if len(sku_data) < 2:
                continue  # Need at least 2 suppliers to compare
            
            # Normalize metrics for comparison (0-1 scale)
            sku_data = sku_data.copy()
            
            # Cost (lower is better)
            min_cost = sku_data['unit_cost'].min()
            max_cost = sku_data['unit_cost'].max()
            if max_cost > min_cost:
                sku_data['cost_score'] = 1 - (sku_data['unit_cost'] - min_cost) / (max_cost - min_cost)
            else:
                sku_data['cost_score'] = 1.0
            
            # Lead time (lower is better)
            min_lead_time = sku_data['lead_time_days'].min()
            max_lead_time = sku_data['lead_time_days'].max()
            if max_lead_time > min_lead_time:
                sku_data['lead_time_score'] = 1 - (sku_data['lead_time_days'] - min_lead_time) / (max_lead_time - min_lead_time)
            else:
                sku_data['lead_time_score'] = 1.0
            
            # Quality (higher is better)
            sku_data['quality_score'] = sku_data['quality_rating'] / 5.0
            
            # Calculate composite score
            sku_data['composite_score'] = (
                sku_data['cost_score'] * criteria['cost_weight'] +
                sku_data['quality_score'] * criteria['quality_weight'] +
                sku_data['lead_time_score'] * criteria['lead_time_weight']
            )
            
            # Rank suppliers
            sku_data = sku_data.sort_values('composite_score', ascending=False)
            sku_data['rank'] = range(1, len(sku_data) + 1)
            
            # Add to results
            for _, row in sku_data.iterrows():
                comparison_results.append({
                    'sku': row['sku'],
                    'supplier_id': row['supplier_id'],
                    'rank': row['rank'],
                    'composite_score': round(row['composite_score'], 3),
                    'cost_score': round(row['cost_score'], 3),
                    'quality_score': round(row['quality_score'], 3),
                    'lead_time_score': round(row['lead_time_score'], 3),
                    'unit_cost': row['unit_cost'],
                    'lead_time_days': row['lead_time_days'],
                    'quality_rating': row['quality_rating'],
                    'recommended': row['rank'] == 1
                })
        
        results_df = pd.DataFrame(comparison_results)
        
        return {
            'success': True,
            'comparison_results': results_df,
            'criteria_used': criteria,
            'skus_analyzed': len(data['sku'].unique()),
            'suppliers_analyzed': len(data['supplier_id'].unique()),
            'recommendations': results_df[results_df['recommended'] == True]
        }
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    def _validate_dataframe_columns(self, df: pd.DataFrame, required_columns: List[str]):
        """Validate that DataFrame contains required columns."""
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
    
    def _apply_procurement_optimization(
        self, 
        base_quantity: float, 
        row: pd.Series, 
        params: Dict[str, Any]
    ) -> float:
        """Apply optimization adjustments to base procurement quantity."""
        # Apply demand variability buffer
        variability_buffer = base_quantity * params['demand_variability']
        adjusted_quantity = base_quantity + variability_buffer
        
        # Apply lead time buffer
        lead_time_adjustment = adjusted_quantity * (params['lead_time_buffer'] - 1)
        final_quantity = adjusted_quantity + lead_time_adjustment
        
        # Ensure minimum order quantity constraint
        return max(final_quantity, row['min_order_qty'])
    
    def _calculate_confidence_score(
        self, 
        row: pd.Series, 
        params: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for recommendations."""
        # Base confidence from service level
        base_confidence = params.get('service_level', 0.95)
        
        # Adjust based on data quality factors
        demand_ratio = row.get('forecast_demand_4weeks', 0) / max(row.get('current_inventory', 1), 1)
        
        # Higher confidence for moderate demand ratios
        if 0.5 <= demand_ratio <= 2.0:
            demand_confidence = 1.0
        elif demand_ratio < 0.5:
            demand_confidence = 0.8
        else:
            demand_confidence = max(0.6, 1.0 / (demand_ratio - 1.0))
        
        return min(1.0, base_confidence * demand_confidence)
    
    # =============================================================================
    # DATA FORMAT CONVERSION UTILITIES
    # =============================================================================
    
    def from_csv(self, csv_data: str, data_type: str = 'procurement') -> pd.DataFrame:
        """
        Convert CSV string to DataFrame for processing.
        
        Args:
            csv_data (str): CSV data as string
            data_type (str): Type of data ('procurement' or 'manufacturing')
            
        Returns:
            pd.DataFrame: Parsed DataFrame
        """
        from io import StringIO
        
        try:
            df = pd.read_csv(StringIO(csv_data))
            return df
        except Exception as e:
            raise ValueError(f"Invalid CSV format: {str(e)}")
    
    def to_csv(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to CSV string.
        
        Args:
            df (pd.DataFrame): DataFrame to convert
            
        Returns:
            str: CSV string
        """
        return df.to_csv(index=False)
    
    def from_json(self, json_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert JSON data to DataFrame for processing.
        
        Args:
            json_data (List[Dict[str, Any]]): List of dictionaries
            
        Returns:
            pd.DataFrame: Converted DataFrame
        """
        return pd.DataFrame(json_data)
    
    def to_json(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to JSON format.
        
        Args:
            df (pd.DataFrame): DataFrame to convert
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries
        """
        return df.to_dict('records')