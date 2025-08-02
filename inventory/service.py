"""
Inventory Domain Service Layer

This module provides the main service interface for inventory management,
including safety stock calculations and reorder point optimization.
All functions work with pandas DataFrames as the primary data format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .models import (
    InventoryStatusInput, InventoryRecommendation,
    RecommendedAction, SafetyStockCalculation
)
from utils.ml_utils import (
    calculate_safety_stock, calculate_reorder_point,
    optimize_inventory_levels
)

logger = logging.getLogger(__name__)


class InventoryService:
    """
    Main service class for inventory management operations.
    
    This service handles safety stock calculations, reorder point optimization,
    and inventory recommendations using pandas DataFrames as the primary data format.
    """
    
    def __init__(self):
        """Initialize the inventory service."""
        self.logger = logging.getLogger(__name__)
    
    # =============================================================================
    # CORE INVENTORY OPTIMIZATION
    # =============================================================================
    
    def calculate_safety_stock(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal safety stock levels and reorder points.
        
        Args:
            data (pd.DataFrame): Inventory data with required columns:
                - sku: Stock Keeping Unit identifier
                - current_stock: Current stock level
                - lead_time_days: Lead time in days
                - service_level_target: Target service level (0.0-1.0)
                - avg_weekly_demand: Average weekly demand
                - demand_std_dev: Standard deviation of demand
            params (Dict[str, Any], optional): Calculation parameters
                
        Returns:
            Dict[str, Any]: Safety stock recommendations with DataFrame
        """
        # Validate input data
        required_columns = [
            'sku', 'current_stock', 'lead_time_days', 
            'service_level_target', 'avg_weekly_demand', 'demand_std_dev'
        ]
        self._validate_dataframe_columns(data, required_columns)
        
        # Set default parameters
        if params is None:
            params = {}
        
        calculation_params = {
            'method': params.get('method', 'statistical'),
            'confidence_level': params.get('confidence_level', 0.95),
            'demand_forecast_accuracy': params.get('demand_forecast_accuracy', 0.85),
            'supply_variability': params.get('supply_variability', 0.10)
        }
        
        # Process each SKU
        recommendations = []
        processing_start = datetime.now()
        
        for _, row in data.iterrows():
            try:
                # Convert weekly demand to daily
                daily_demand = row['avg_weekly_demand'] / 7
                daily_std_dev = row['demand_std_dev'] / 7
                
                # Calculate safety stock using ML utils
                safety_stock = calculate_safety_stock(
                    demand_mean=daily_demand,
                    demand_std=daily_std_dev,
                    lead_time=row['lead_time_days'],
                    service_level=row['service_level_target']
                )
                
                # Calculate reorder point
                reorder_point = calculate_reorder_point(
                    demand_mean=daily_demand,
                    lead_time=row['lead_time_days'],
                    safety_stock=safety_stock
                )
                
                # Determine recommended action
                action, days_until_stockout = self._determine_inventory_action(
                    current_stock=row['current_stock'],
                    reorder_point=reorder_point,
                    safety_stock=safety_stock,
                    daily_demand=daily_demand
                )
                
                # Calculate confidence score
                confidence_score = self._calculate_inventory_confidence(
                    row, calculation_params
                )
                
                # Create calculation details
                calculation_details = {
                    'method': calculation_params['method'],
                    'service_level_achieved': row['service_level_target'],
                    'lead_time_demand': daily_demand * row['lead_time_days'],
                    'safety_factor': safety_stock / (daily_std_dev * np.sqrt(row['lead_time_days'])) if daily_std_dev > 0 else 0
                }
                
                recommendation = {
                    'sku': row['sku'],
                    'safety_stock': int(round(safety_stock)),
                    'reorder_point': int(round(reorder_point)),
                    'current_stock': int(row['current_stock']),
                    'recommended_action': action.value,
                    'days_until_stockout': days_until_stockout,
                    'confidence_score': round(confidence_score, 3),
                    'calculation_details': calculation_details
                }
                
                recommendations.append(recommendation)
                
            except Exception as e:
                self.logger.error(f"Error processing SKU {row['sku']}: {str(e)}")
                continue
        
        # Create results DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        # Calculate summary metrics
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        # Calculate total safety stock value (assuming unit cost if available)
        total_safety_stock_value = 0
        if 'unit_cost' in data.columns:
            for _, rec in recommendations_df.iterrows():
                unit_cost = data[data['sku'] == rec['sku']]['unit_cost'].iloc[0]
                total_safety_stock_value += rec['safety_stock'] * unit_cost
        
        return {
            'success': True,
            'recommendations': recommendations_df,
            'recommendations_count': len(recommendations),
            'processing_time_seconds': round(processing_time, 3),
            'optimization_summary': {
                'total_items_analyzed': len(recommendations),
                'items_needing_reorder': len(recommendations_df[
                    recommendations_df['recommended_action'].isin(['REORDER', 'URGENT_REORDER'])
                ]),
                'avg_service_level': data['service_level_target'].mean(),
                'total_safety_stock_value': round(total_safety_stock_value, 2),
                'urgent_reorders': len(recommendations_df[
                    recommendations_df['recommended_action'] == 'URGENT_REORDER'
                ]),
                'sufficient_stock_items': len(recommendations_df[
                    recommendations_df['recommended_action'] == 'SUFFICIENT_STOCK'
                ])
            }
        }
    
    def analyze_abc_classification(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform ABC analysis for inventory prioritization.
        
        Args:
            data (pd.DataFrame): Inventory data with columns:
                - sku: Stock Keeping Unit identifier
                - annual_usage_value: Annual usage value
                - current_stock: Current stock level
                - unit_cost: Unit cost
            params (Dict[str, Any], optional): Classification parameters
                
        Returns:
            Dict[str, Any]: ABC classification results
        """
        required_columns = ['sku', 'annual_usage_value']
        self._validate_dataframe_columns(data, required_columns)
        
        if params is None:
            params = {}
        
        # Classification thresholds
        a_threshold = params.get('a_threshold', 0.8)  # Top 80% of value
        b_threshold = params.get('b_threshold', 0.95)  # Next 15% of value
        
        # Calculate cumulative percentage
        df = data.copy()
        df = df.sort_values('annual_usage_value', ascending=False)
        df['cumulative_value'] = df['annual_usage_value'].cumsum()
        total_value = df['annual_usage_value'].sum()
        df['cumulative_percentage'] = df['cumulative_value'] / total_value
        
        # Assign ABC classification
        df['abc_class'] = 'C'
        df.loc[df['cumulative_percentage'] <= a_threshold, 'abc_class'] = 'A'
        df.loc[(df['cumulative_percentage'] > a_threshold) & 
               (df['cumulative_percentage'] <= b_threshold), 'abc_class'] = 'B'
        
        # Calculate class statistics
        agg_dict = {
            'sku': 'count',
            'annual_usage_value': ['sum', 'mean']
        }
        
        if 'current_stock' in df.columns:
            agg_dict['current_stock'] = 'sum'
        
        class_stats = df.groupby('abc_class').agg(agg_dict).round(2)
        
        return {
            'success': True,
            'classification_results': df,
            'class_statistics': class_stats,
            'classification_params': {
                'a_threshold': a_threshold,
                'b_threshold': b_threshold
            },
            'summary': {
                'total_items': len(df),
                'class_a_items': len(df[df['abc_class'] == 'A']),
                'class_b_items': len(df[df['abc_class'] == 'B']),
                'class_c_items': len(df[df['abc_class'] == 'C']),
                'total_value': total_value
            }
        }
    
    def generate_stock_alerts(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate stock level alerts and recommendations.
        
        Args:
            data (pd.DataFrame): Current stock data with columns:
                - sku: Stock Keeping Unit identifier
                - current_stock: Current stock level
                - reorder_point: Reorder point
                - safety_stock: Safety stock level
            params (Dict[str, Any], optional): Alert parameters
                
        Returns:
            Dict[str, Any]: Stock alerts and recommendations
        """
        required_columns = ['sku', 'current_stock', 'reorder_point', 'safety_stock']
        self._validate_dataframe_columns(data, required_columns)
        
        if params is None:
            params = {}
        
        # Alert thresholds
        urgent_threshold = params.get('urgent_threshold', 0.5)  # 50% of safety stock
        warning_threshold = params.get('warning_threshold', 1.0)  # At safety stock level
        
        alerts = []
        
        for _, row in data.iterrows():
            alert_type = None
            priority = 'LOW'
            message = ''
            
            if row['current_stock'] <= row['safety_stock'] * urgent_threshold:
                alert_type = 'URGENT_REORDER'
                priority = 'HIGH'
                message = f"Critical stock level: {row['current_stock']} units remaining"
            elif row['current_stock'] <= row['reorder_point']:
                alert_type = 'REORDER'
                priority = 'MEDIUM'
                message = f"Below reorder point: {row['current_stock']} units remaining"
            elif row['current_stock'] <= row['safety_stock'] * warning_threshold:
                alert_type = 'WARNING'
                priority = 'LOW'
                message = f"Approaching safety stock level: {row['current_stock']} units remaining"
            else:
                alert_type = 'SUFFICIENT_STOCK'
                priority = 'LOW'
                message = f"Stock level adequate: {row['current_stock']} units available"
            
            alerts.append({
                'sku': row['sku'],
                'alert_type': alert_type,
                'priority': priority,
                'message': message,
                'current_stock': int(row['current_stock']),
                'reorder_point': int(row['reorder_point']),
                'safety_stock': int(row['safety_stock']),
                'stock_coverage_days': self._calculate_stock_coverage(row),
                'timestamp': datetime.now().isoformat()
            })
        
        alerts_df = pd.DataFrame(alerts)
        
        return {
            'success': True,
            'alerts': alerts_df,
            'alert_summary': {
                'total_items': len(alerts),
                'urgent_alerts': len(alerts_df[alerts_df['alert_type'] == 'URGENT_REORDER']),
                'reorder_alerts': len(alerts_df[alerts_df['alert_type'] == 'REORDER']),
                'warning_alerts': len(alerts_df[alerts_df['alert_type'] == 'WARNING']),
                'sufficient_stock': len(alerts_df[alerts_df['alert_type'] == 'SUFFICIENT_STOCK'])
            }
        }
    
    def calculate_turnover_metrics(
        self, 
        data: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate inventory turnover metrics.
        
        Args:
            data (pd.DataFrame): Inventory data with columns:
                - sku: Stock Keeping Unit identifier
                - annual_demand: Annual demand quantity
                - avg_inventory_value: Average inventory value
            params (Dict[str, Any], optional): Calculation parameters
                
        Returns:
            Dict[str, Any]: Turnover metrics
        """
        required_columns = ['sku', 'annual_demand', 'avg_inventory_value']
        self._validate_dataframe_columns(data, required_columns)
        
        df = data.copy()
        
        # Calculate turnover ratio
        df['turnover_ratio'] = df['annual_demand'] / df['avg_inventory_value'].replace(0, 1)
        
        # Calculate days of inventory
        df['days_of_inventory'] = 365 / df['turnover_ratio'].replace(0, 365)
        
        # Classify turnover performance
        df['turnover_class'] = 'SLOW'
        df.loc[df['turnover_ratio'] >= 4, 'turnover_class'] = 'FAST'
        df.loc[(df['turnover_ratio'] >= 2) & (df['turnover_ratio'] < 4), 'turnover_class'] = 'MEDIUM'
        
        # Calculate summary statistics
        summary_stats = {
            'avg_turnover_ratio': df['turnover_ratio'].mean(),
            'median_turnover_ratio': df['turnover_ratio'].median(),
            'avg_days_of_inventory': df['days_of_inventory'].mean(),
            'fast_movers': len(df[df['turnover_class'] == 'FAST']),
            'medium_movers': len(df[df['turnover_class'] == 'MEDIUM']),
            'slow_movers': len(df[df['turnover_class'] == 'SLOW']),
            'total_inventory_value': df['avg_inventory_value'].sum()
        }
        
        return {
            'success': True,
            'turnover_metrics': df,
            'summary_statistics': summary_stats,
            'recommendations': self._generate_turnover_recommendations(df)
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
    
    def _determine_inventory_action(
        self, 
        current_stock: float, 
        reorder_point: float, 
        safety_stock: float, 
        daily_demand: float
    ) -> tuple:
        """Determine recommended inventory action and days until stockout."""
        if current_stock <= safety_stock * 0.5:
            action = RecommendedAction.URGENT_REORDER
            days_until_stockout = max(1, int(current_stock / max(daily_demand, 1)))
        elif current_stock <= reorder_point:
            action = RecommendedAction.REORDER
            days_until_stockout = max(7, int((current_stock - safety_stock) / max(daily_demand, 1)))
        elif current_stock > reorder_point * 2:
            action = RecommendedAction.EXCESS_STOCK
            days_until_stockout = int(current_stock / max(daily_demand, 1))
        else:
            action = RecommendedAction.SUFFICIENT_STOCK
            days_until_stockout = int((current_stock - safety_stock) / max(daily_demand, 1))
        
        return action, max(0, days_until_stockout)
    
    def _calculate_inventory_confidence(
        self, 
        row: pd.Series, 
        params: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for inventory recommendations."""
        base_confidence = params.get('confidence_level', 0.95)
        
        # Adjust based on demand variability
        cv = row['demand_std_dev'] / max(row['avg_weekly_demand'], 1)
        if cv < 0.2:
            variability_factor = 1.0
        elif cv < 0.5:
            variability_factor = 0.9
        else:
            variability_factor = 0.8
        
        # Adjust based on lead time
        if row['lead_time_days'] <= 7:
            lead_time_factor = 1.0
        elif row['lead_time_days'] <= 21:
            lead_time_factor = 0.95
        else:
            lead_time_factor = 0.9
        
        return min(1.0, base_confidence * variability_factor * lead_time_factor)
    
    def _calculate_stock_coverage(self, row: pd.Series) -> int:
        """Calculate days of stock coverage."""
        if 'avg_weekly_demand' in row and row['avg_weekly_demand'] > 0:
            daily_demand = row['avg_weekly_demand'] / 7
            return int(row['current_stock'] / daily_demand)
        return 0
    
    def _generate_turnover_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate recommendations based on turnover analysis."""
        recommendations = []
        
        # Recommendations for slow movers
        slow_movers = df[df['turnover_class'] == 'SLOW']
        if not slow_movers.empty:
            recommendations.append({
                'category': 'SLOW_MOVERS',
                'count': len(slow_movers),
                'recommendation': 'Consider reducing safety stock levels and order quantities',
                'potential_savings': slow_movers['avg_inventory_value'].sum() * 0.2
            })
        
        # Recommendations for fast movers
        fast_movers = df[df['turnover_class'] == 'FAST']
        if not fast_movers.empty:
            recommendations.append({
                'category': 'FAST_MOVERS',
                'count': len(fast_movers),
                'recommendation': 'Monitor closely for stockouts and consider increasing safety stock',
                'risk_level': 'HIGH'
            })
        
        return recommendations
    
    # =============================================================================
    # DATA FORMAT CONVERSION UTILITIES
    # =============================================================================
    
    def from_csv(self, csv_data: str) -> pd.DataFrame:
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