"""
Dummy data generation utilities for testing and development.

This module provides realistic CSV data generation matching exact input/output
formats for all domain modules in the Balancer platform.

Key Features:
- Realistic data generation for all domain modules
- Reproducible data with configurable seeds
- CSV format matching exact specifications
- Comprehensive test data for development and testing
- Support for time series data with trends and seasonality

Supported Domains:
- Supply: Procurement and manufacturing data
- Inventory: Stock levels and recommendations
- Demand: Time series demand data with patterns
- Distribution: Orders, vehicles, and route optimization

Author: Balancer Platform
Version: 1.0.0
"""

import csv
import io
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker with default seed for reproducibility
fake = Faker()
random.seed(42)  # For reproducible test data
np.random.seed(42)


# =============================================================================
# MAIN DATA GENERATOR CLASS
# =============================================================================

class DummyDataGenerator:
    """
    Generator for realistic dummy data across all domain modules.
    
    This class provides methods to generate realistic test data for all
    domain modules in the Balancer platform, ensuring data consistency
    and reproducibility through seeded random generation.
    
    Attributes:
        seed (int): Random seed for reproducible data generation
        sku_prefixes (List[str]): Prefixes for SKU generation
        product_categories (List[str]): Product category codes
        supplier_names (List[str]): Realistic supplier company names
        cities (List[Dict]): Geographic locations with coordinates
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the dummy data generator with a seed for reproducibility.
        
        Args:
            seed (int): Random seed for consistent data generation across runs
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        fake.seed_instance(seed)
        
        # Common SKU patterns
        self.sku_prefixes = ['SKU', 'PROD', 'ITEM', 'PART']
        self.product_categories = ['ELEC', 'MECH', 'CHEM', 'TEXT', 'FOOD']
        
        # Supplier data
        self.supplier_names = [
            'GlobalSupply Corp', 'MegaManufacturing Ltd', 'PrimeParts Inc',
            'QualityComponents Co', 'ReliableSuppliers LLC', 'FastDelivery Corp',
            'BulkGoods Industries', 'PrecisionParts Ltd', 'EcoFriendly Supplies',
            'TechComponents Inc'
        ]
        
        # Location data for distribution
        self.cities = [
            {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
            {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
            {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
            {'name': 'Houston', 'lat': 29.7604, 'lon': -95.3698},
            {'name': 'Phoenix', 'lat': 33.4484, 'lon': -112.0740},
            {'name': 'Philadelphia', 'lat': 39.9526, 'lon': -75.1652},
            {'name': 'San Antonio', 'lat': 29.4241, 'lon': -98.4936},
            {'name': 'San Diego', 'lat': 32.7157, 'lon': -117.1611},
            {'name': 'Dallas', 'lat': 32.7767, 'lon': -96.7970},
            {'name': 'San Jose', 'lat': 37.3382, 'lon': -121.8863}
        ]
    
    # =========================================================================
    # UTILITY METHODS FOR ID GENERATION
    # =========================================================================
    
    def generate_sku(self) -> str:
        """
        Generate a realistic Stock Keeping Unit (SKU) identifier.
        
        Returns:
            str: SKU in format PREFIX-CATEGORY-NUMBER (e.g., "SKU-ELEC-1234")
        """
        prefix = random.choice(self.sku_prefixes)
        category = random.choice(self.product_categories)
        number = random.randint(1000, 9999)
        return f"{prefix}-{category}-{number}"
    
    def generate_supplier_id(self) -> str:
        """Generate a realistic supplier ID."""
        return f"SUP-{random.randint(100, 999)}"
    
    def generate_vehicle_id(self) -> str:
        """Generate a realistic vehicle ID."""
        return f"VEH-{random.randint(1000, 9999)}"
    
    def generate_order_id(self) -> str:
        """Generate a realistic order ID."""
        return f"ORD-{random.randint(10000, 99999)}"
    
    # =========================================================================
    # SUPPLY DOMAIN DATA GENERATION
    # =========================================================================
    
    def generate_procurement_data(self, num_records: int = 50) -> List[Dict[str, Any]]:
        """
        Generate procurement data CSV format.
        
        Columns: sku, current_inventory, forecast_demand_4weeks, safety_stock, 
                min_order_qty, supplier_id, unit_cost
        """
        data = []
        
        for _ in range(num_records):
            forecast_demand = random.randint(100, 2000)
            current_inventory = random.randint(0, int(forecast_demand * 1.5))
            safety_stock = int(forecast_demand * random.uniform(0.1, 0.3))
            
            record = {
                'sku': self.generate_sku(),
                'current_inventory': current_inventory,
                'forecast_demand_4weeks': forecast_demand,
                'safety_stock': safety_stock,
                'min_order_qty': random.randint(50, 500),
                'supplier_id': self.generate_supplier_id(),
                'unit_cost': round(random.uniform(5.0, 500.0), 2)
            }
            data.append(record)
        
        return data
    
    def generate_manufacturing_data(self, num_records: int = 50) -> List[Dict[str, Any]]:
        """
        Generate manufacturing data CSV format.
        
        Columns: sku, current_inventory, forecast_demand_4weeks, safety_stock,
                batch_size, production_time_days, unit_cost
        """
        data = []
        
        for _ in range(num_records):
            forecast_demand = random.randint(100, 2000)
            current_inventory = random.randint(0, int(forecast_demand * 1.5))
            safety_stock = int(forecast_demand * random.uniform(0.1, 0.3))
            
            record = {
                'sku': self.generate_sku(),
                'current_inventory': current_inventory,
                'forecast_demand_4weeks': forecast_demand,
                'safety_stock': safety_stock,
                'batch_size': random.randint(100, 1000),
                'production_time_days': random.randint(1, 14),
                'unit_cost': round(random.uniform(3.0, 300.0), 2)
            }
            data.append(record)
        
        return data
    
    def generate_procurement_recommendations(self, num_records: int = 50) -> List[Dict[str, Any]]:
        """
        Generate procurement recommendations CSV format.
        
        Columns: sku, recommended_quantity, supplier_id, order_date, 
                expected_delivery, total_cost
        """
        data = []
        base_date = datetime.now()
        
        for _ in range(num_records):
            order_date = base_date + timedelta(days=random.randint(0, 7))
            delivery_date = order_date + timedelta(days=random.randint(3, 21))
            quantity = random.randint(100, 2000)
            unit_cost = random.uniform(5.0, 500.0)
            
            record = {
                'sku': self.generate_sku(),
                'recommended_quantity': quantity,
                'supplier_id': self.generate_supplier_id(),
                'order_date': order_date.strftime('%Y-%m-%d'),
                'expected_delivery': delivery_date.strftime('%Y-%m-%d'),
                'total_cost': round(quantity * unit_cost, 2)
            }
            data.append(record)
        
        return data
    
    def generate_manufacturing_recommendations(self, num_records: int = 50) -> List[Dict[str, Any]]:
        """
        Generate manufacturing recommendations CSV format.
        
        Columns: sku, recommended_batch_qty, production_start_date,
                production_complete_date, total_cost
        """
        data = []
        base_date = datetime.now()
        
        for _ in range(num_records):
            start_date = base_date + timedelta(days=random.randint(0, 7))
            production_days = random.randint(1, 14)
            complete_date = start_date + timedelta(days=production_days)
            batch_qty = random.randint(100, 1000)
            unit_cost = random.uniform(3.0, 300.0)
            
            record = {
                'sku': self.generate_sku(),
                'recommended_batch_qty': batch_qty,
                'production_start_date': start_date.strftime('%Y-%m-%d'),
                'production_complete_date': complete_date.strftime('%Y-%m-%d'),
                'total_cost': round(batch_qty * unit_cost, 2)
            }
            data.append(record)
        
        return data
    
    # =========================================================================
    # INVENTORY DOMAIN DATA GENERATION
    # =========================================================================
    
    def generate_inventory_status(self, num_records: int = 50) -> List[Dict[str, Any]]:
        """
        Generate inventory status CSV format.
        
        Columns: sku, current_stock, lead_time_days, service_level_target,
                avg_weekly_demand, demand_std_dev
        """
        data = []
        
        for _ in range(num_records):
            avg_demand = random.uniform(50, 500)
            std_dev = avg_demand * random.uniform(0.1, 0.4)  # 10-40% CV
            current_stock = random.randint(0, int(avg_demand * 8))  # 0-8 weeks of stock
            
            record = {
                'sku': self.generate_sku(),
                'current_stock': current_stock,
                'lead_time_days': random.randint(1, 28),
                'service_level_target': round(random.uniform(0.90, 0.99), 2),
                'avg_weekly_demand': round(avg_demand, 2),
                'demand_std_dev': round(std_dev, 2)
            }
            data.append(record)
        
        return data
    
    def generate_inventory_recommendations(self, num_records: int = 50) -> List[Dict[str, Any]]:
        """
        Generate inventory recommendations CSV format.
        
        Columns: sku, safety_stock, reorder_point, current_stock,
                recommended_action, days_until_stockout
        """
        data = []
        actions = ['REORDER', 'URGENT_REORDER', 'SUFFICIENT_STOCK']
        
        for _ in range(num_records):
            current_stock = random.randint(0, 2000)
            safety_stock = random.randint(50, 500)
            reorder_point = safety_stock + random.randint(100, 300)
            
            # Determine action based on stock levels
            if current_stock <= safety_stock:
                action = 'URGENT_REORDER'
                days_until_stockout = random.randint(1, 7)
            elif current_stock <= reorder_point:
                action = 'REORDER'
                days_until_stockout = random.randint(8, 21)
            else:
                action = 'SUFFICIENT_STOCK'
                days_until_stockout = random.randint(22, 90)
            
            record = {
                'sku': self.generate_sku(),
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'current_stock': current_stock,
                'recommended_action': action,
                'days_until_stockout': days_until_stockout
            }
            data.append(record)
        
        return data
    
    # =========================================================================
    # DEMAND DOMAIN DATA GENERATION
    # =========================================================================
    
    def generate_demand_data(self, num_records: int = 500, num_skus: int = 20) -> List[Dict[str, Any]]:
        """
        Generate demand data CSV format with time series.
        
        Columns: date, sku, quantity
        """
        data = []
        skus = [self.generate_sku() for _ in range(num_skus)]
        
        # Generate data for the last 52 weeks (1 year)
        start_date = datetime.now() - timedelta(weeks=52)
        
        for sku in skus:
            # Base demand with trend and seasonality
            base_demand = random.uniform(100, 1000)
            trend = random.uniform(-0.5, 0.5)  # Weekly trend
            
            for week in range(52):
                date = start_date + timedelta(weeks=week)
                
                # Add seasonality (annual cycle)
                seasonality = 0.2 * np.sin(2 * np.pi * week / 52)
                
                # Add trend
                trend_component = trend * week
                
                # Add noise
                noise = random.uniform(-0.2, 0.2)
                
                # Calculate demand
                demand = base_demand * (1 + seasonality + trend_component + noise)
                demand = max(0, int(demand))  # Ensure non-negative
                
                record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'sku': sku,
                    'quantity': demand
                }
                data.append(record)
        
        return data
    
    def generate_forecast_output(self, num_records: int = 100) -> List[Dict[str, Any]]:
        """
        Generate forecast output CSV format.
        
        Columns: sku, forecast_date, prediction
        """
        data = []
        skus = [self.generate_sku() for _ in range(20)]
        
        # Generate forecasts for next 12 weeks
        start_date = datetime.now()
        
        for sku in skus:
            base_prediction = random.uniform(100, 1000)
            
            for week in range(12):
                forecast_date = start_date + timedelta(weeks=week)
                
                # Add some variation to predictions
                variation = random.uniform(0.8, 1.2)
                prediction = base_prediction * variation
                
                record = {
                    'sku': sku,
                    'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                    'prediction': round(prediction, 2)
                }
                data.append(record)
                
                if len(data) >= num_records:
                    break
            
            if len(data) >= num_records:
                break
        
        return data[:num_records]
    
    # =========================================================================
    # DISTRIBUTION DOMAIN DATA GENERATION
    # =========================================================================
    
    def generate_orders_and_vehicles(self, num_orders: int = 50, 
                                   num_vehicles: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate orders and vehicles data for distribution optimization.
        
        Returns dictionary with 'orders' and 'vehicles' keys.
        """
        # Generate orders
        orders = []
        for _ in range(num_orders):
            city = random.choice(self.cities)
            # Add some random variation to coordinates
            lat_variation = random.uniform(-0.1, 0.1)
            lon_variation = random.uniform(-0.1, 0.1)
            
            order = {
                'order_id': self.generate_order_id(),
                'customer_lat': round(city['lat'] + lat_variation, 6),
                'customer_lon': round(city['lon'] + lon_variation, 6),
                'volume_m3': round(random.uniform(0.1, 5.0), 2),
                'weight_kg': round(random.uniform(1.0, 100.0), 2)
            }
            orders.append(order)
        
        # Generate vehicles
        vehicles = []
        for _ in range(num_vehicles):
            vehicle = {
                'vehicle_id': self.generate_vehicle_id(),
                'max_volume_m3': round(random.uniform(10.0, 50.0), 2),
                'max_weight_kg': round(random.uniform(500.0, 2000.0), 2),
                'cost_per_km': round(random.uniform(0.5, 2.0), 2)
            }
            vehicles.append(vehicle)
        
        return {'orders': orders, 'vehicles': vehicles}
    
    def generate_route_assignments(self, num_records: int = 50) -> List[Dict[str, Any]]:
        """
        Generate route assignments CSV format.
        
        Columns: vehicle_id, order_id, sequence, distance_km, total_cost
        """
        data = []
        vehicles = [self.generate_vehicle_id() for _ in range(10)]
        
        for _ in range(num_records):
            distance = random.uniform(5.0, 200.0)
            cost_per_km = random.uniform(0.5, 2.0)
            
            record = {
                'vehicle_id': random.choice(vehicles),
                'order_id': self.generate_order_id(),
                'sequence': random.randint(1, 10),
                'distance_km': round(distance, 2),
                'total_cost': round(distance * cost_per_km, 2)
            }
            data.append(record)
        
        return data
    
    # =========================================================================
    # CSV GENERATION AND FILE UTILITIES
    # =========================================================================
    
    def generate_csv_string(self, data: List[Dict[str, Any]]) -> str:
        """Convert data list to CSV string format."""
        if not data:
            return ""
        
        import io
        import csv
        
        output = io.StringIO()
        fieldnames = data[0].keys()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        
        return output.getvalue()
    
    def save_sample_data(self, output_dir: str = "sample_data") -> None:
        """Generate and save sample CSV files for all domains."""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Supply domain samples
        procurement_data = self.generate_procurement_data()
        with open(f"{output_dir}/procurement_data.csv", 'w') as f:
            f.write(self.generate_csv_string(procurement_data))
        
        manufacturing_data = self.generate_manufacturing_data()
        with open(f"{output_dir}/manufacturing_data.csv", 'w') as f:
            f.write(self.generate_csv_string(manufacturing_data))
        
        # Inventory domain samples
        inventory_status = self.generate_inventory_status()
        with open(f"{output_dir}/inventory_status.csv", 'w') as f:
            f.write(self.generate_csv_string(inventory_status))
        
        # Demand domain samples
        demand_data = self.generate_demand_data()
        with open(f"{output_dir}/demand_data.csv", 'w') as f:
            f.write(self.generate_csv_string(demand_data))
        
        # Distribution domain samples
        orders_vehicles = self.generate_orders_and_vehicles()
        
        # Combine orders and vehicles into single CSV for simplicity
        combined_data = []
        for order in orders_vehicles['orders']:
            combined_data.append({**order, 'type': 'order'})
        for vehicle in orders_vehicles['vehicles']:
            combined_data.append({**vehicle, 'type': 'vehicle'})
        
        with open(f"{output_dir}/orders_and_vehicles.csv", 'w') as f:
            f.write(self.generate_csv_string(combined_data))
        
        print(f"Sample data files generated in {output_dir}/ directory")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR QUICK DATA GENERATION
# =============================================================================

def get_sample_procurement_data(num_records: int = 50) -> List[Dict[str, Any]]:
    """Get sample procurement data."""
    generator = DummyDataGenerator()
    return generator.generate_procurement_data(num_records)


def get_sample_manufacturing_data(num_records: int = 50) -> List[Dict[str, Any]]:
    """Get sample manufacturing data."""
    generator = DummyDataGenerator()
    return generator.generate_manufacturing_data(num_records)


def get_sample_inventory_data(num_records: int = 50) -> List[Dict[str, Any]]:
    """Get sample inventory status data."""
    generator = DummyDataGenerator()
    return generator.generate_inventory_status(num_records)


def get_sample_demand_data(num_records: int = 500, num_skus: int = 20) -> List[Dict[str, Any]]:
    """Get sample demand data with time series."""
    generator = DummyDataGenerator()
    return generator.generate_demand_data(num_records, num_skus)


def get_sample_distribution_data(num_orders: int = 50, 
                               num_vehicles: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """Get sample distribution data."""
    generator = DummyDataGenerator()
    return generator.generate_orders_and_vehicles(num_orders, num_vehicles)


def get_sample_csv_string(data_type: str, **kwargs) -> str:
    """
    Get sample data as CSV string.
    
    Args:
        data_type: Type of data ('procurement', 'manufacturing', 'inventory', 
                  'demand', 'distribution')
        **kwargs: Additional arguments for data generation
    
    Returns:
        CSV string of sample data
    """
    generator = DummyDataGenerator()
    
    if data_type == 'procurement':
        data = generator.generate_procurement_data(kwargs.get('num_records', 50))
    elif data_type == 'manufacturing':
        data = generator.generate_manufacturing_data(kwargs.get('num_records', 50))
    elif data_type == 'inventory':
        data = generator.generate_inventory_status(kwargs.get('num_records', 50))
    elif data_type == 'demand':
        data = generator.generate_demand_data(
            kwargs.get('num_records', 500), 
            kwargs.get('num_skus', 20)
        )
    elif data_type == 'distribution':
        orders_vehicles = generator.generate_orders_and_vehicles(
            kwargs.get('num_orders', 50),
            kwargs.get('num_vehicles', 10)
        )
        # Return orders CSV for simplicity
        data = orders_vehicles['orders']
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    return generator.generate_csv_string(data)


# =============================================================================
# WRAPPER FUNCTIONS FOR TEST COMPATIBILITY
# =============================================================================

def generate_procurement_data(num_records: int = 50) -> List[Dict[str, Any]]:
    """Generate procurement data for testing."""
    generator = DummyDataGenerator()
    return generator.generate_procurement_data(num_records)


def generate_manufacturing_data(num_records: int = 50) -> List[Dict[str, Any]]:
    """Generate manufacturing data for testing."""
    generator = DummyDataGenerator()
    return generator.generate_manufacturing_data(num_records)


def generate_inventory_data(num_records: int = 50) -> List[Dict[str, Any]]:
    """Generate inventory data for testing."""
    generator = DummyDataGenerator()
    return generator.generate_inventory_status(num_records)


def generate_demand_data(num_records: int = 500, num_skus: int = 20) -> List[Dict[str, Any]]:
    """Generate demand data for testing."""
    generator = DummyDataGenerator()
    return generator.generate_demand_data(num_records, num_skus)


def generate_distribution_data(num_orders: int = 50, num_vehicles: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """Generate distribution data for testing."""
    generator = DummyDataGenerator()
    return generator.generate_orders_and_vehicles(num_orders, num_vehicles)


if __name__ == "__main__":
    # Generate sample data files when run directly
    generator = DummyDataGenerator()
    generator.save_sample_data()