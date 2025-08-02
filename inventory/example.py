"""
Inventory Domain Example Usage

This example demonstrates how to use the inventory domain for safety stock
calculations and inventory optimization using the new service layer with DataFrames.
"""

import pandas as pd
from inventory.service import InventoryService
from utils.dummy_data import DummyDataGenerator


def main():
    print("=== Inventory Domain Example ===")
    
    # Initialize service
    inventory_service = InventoryService()
    
    # Generate sample data
    generator = DummyDataGenerator(seed=42)
    
    # Example 1: Safety Stock Calculation
    print("\n1. Safety Stock Calculation Example")
    print("-" * 40)
    
    # Generate sample inventory data
    inventory_data = generator.generate_inventory_status(5)
    inventory_df = pd.DataFrame(inventory_data)
    
    print("Input data:")
    print(inventory_df[['sku', 'current_stock', 'lead_time_days', 'service_level_target', 'avg_weekly_demand']].head())
    
    # Calculate safety stock
    result = inventory_service.calculate_safety_stock(inventory_df)
    
    if result['success']:
        print(f"\nCalculation successful! Generated {result['recommendations_count']} recommendations")
        print(f"Processing time: {result['processing_time_seconds']} seconds")
        print(f"Items needing reorder: {result['optimization_summary']['items_needing_reorder']}")
        
        print("\nRecommendations:")
        recommendations_df = result['recommendations']
        for _, rec in recommendations_df.head().iterrows():
            print(f"  {rec['sku']}: Safety Stock = {rec['safety_stock']}, "
                  f"Reorder Point = {rec['reorder_point']}, Action = {rec['recommended_action']}")
    
    # Example 2: ABC Classification Analysis
    print("\n2. ABC Classification Example")
    print("-" * 40)
    
    # Create sample ABC analysis data
    abc_data = [
        {'sku': 'SKU-001', 'annual_usage_value': 50000, 'current_stock': 200, 'unit_cost': 25.00},
        {'sku': 'SKU-002', 'annual_usage_value': 30000, 'current_stock': 150, 'unit_cost': 20.00},
        {'sku': 'SKU-003', 'annual_usage_value': 25000, 'current_stock': 100, 'unit_cost': 15.00},
        {'sku': 'SKU-004', 'annual_usage_value': 15000, 'current_stock': 80, 'unit_cost': 12.00},
        {'sku': 'SKU-005', 'annual_usage_value': 8000, 'current_stock': 60, 'unit_cost': 10.00},
        {'sku': 'SKU-006', 'annual_usage_value': 5000, 'current_stock': 40, 'unit_cost': 8.00},
        {'sku': 'SKU-007', 'annual_usage_value': 3000, 'current_stock': 30, 'unit_cost': 6.00},
        {'sku': 'SKU-008', 'annual_usage_value': 2000, 'current_stock': 20, 'unit_cost': 4.00}
    ]
    abc_df = pd.DataFrame(abc_data)
    
    print("ABC analysis input data:")
    print(abc_df[['sku', 'annual_usage_value', 'unit_cost']].head())
    
    # Perform ABC analysis
    abc_result = inventory_service.analyze_abc_classification(abc_df)
    
    if abc_result['success']:
        print(f"\nABC Analysis complete!")
        print(f"Total items: {abc_result['summary']['total_items']}")
        print(f"Class A items: {abc_result['summary']['class_a_items']}")
        print(f"Class B items: {abc_result['summary']['class_b_items']}")
        print(f"Class C items: {abc_result['summary']['class_c_items']}")
        
        print("\nClassification results:")
        classification_df = abc_result['classification_results']
        for _, item in classification_df.head().iterrows():
            print(f"  {item['sku']}: Class {item['abc_class']} "
                  f"(Value: ${item['annual_usage_value']:,}, "
                  f"Cumulative: {item['cumulative_percentage']:.1%})")
    
    # Example 3: Stock Alerts Generation
    print("\n3. Stock Alerts Example")
    print("-" * 40)
    
    # Create sample current stock data
    stock_data = [
        {'sku': 'SKU-001', 'current_stock': 45, 'reorder_point': 100, 'safety_stock': 50},
        {'sku': 'SKU-002', 'current_stock': 200, 'reorder_point': 150, 'safety_stock': 75},
        {'sku': 'SKU-003', 'current_stock': 25, 'reorder_point': 80, 'safety_stock': 40},
        {'sku': 'SKU-004', 'current_stock': 300, 'reorder_point': 200, 'safety_stock': 100},
        {'sku': 'SKU-005', 'current_stock': 15, 'reorder_point': 60, 'safety_stock': 30}
    ]
    stock_df = pd.DataFrame(stock_data)
    
    print("Current stock levels:")
    print(stock_df)
    
    # Generate stock alerts
    alerts_result = inventory_service.generate_stock_alerts(stock_df)
    
    if alerts_result['success']:
        print(f"\nAlerts generated!")
        print(f"Total items: {alerts_result['alert_summary']['total_items']}")
        print(f"Urgent alerts: {alerts_result['alert_summary']['urgent_alerts']}")
        print(f"Reorder alerts: {alerts_result['alert_summary']['reorder_alerts']}")
        
        print("\nStock alerts:")
        alerts_df = alerts_result['alerts']
        for _, alert in alerts_df.iterrows():
            if alert['alert_type'] in ['URGENT_REORDER', 'REORDER']:
                print(f"  {alert['sku']}: {alert['alert_type']} - {alert['message']}")
    
    # Example 4: Inventory Turnover Analysis
    print("\n4. Inventory Turnover Analysis Example")
    print("-" * 40)
    
    # Create sample turnover data
    turnover_data = [
        {'sku': 'SKU-001', 'annual_demand': 2400, 'avg_inventory_value': 5000},
        {'sku': 'SKU-002', 'annual_demand': 1800, 'avg_inventory_value': 3000},
        {'sku': 'SKU-003', 'annual_demand': 1200, 'avg_inventory_value': 2000},
        {'sku': 'SKU-004', 'annual_demand': 600, 'avg_inventory_value': 1500},
        {'sku': 'SKU-005', 'annual_demand': 300, 'avg_inventory_value': 1000}
    ]
    turnover_df = pd.DataFrame(turnover_data)
    
    print("Turnover analysis input:")
    print(turnover_df)
    
    # Calculate turnover metrics
    turnover_result = inventory_service.calculate_turnover_metrics(turnover_df)
    
    if turnover_result['success']:
        print(f"\nTurnover analysis complete!")
        
        summary = turnover_result['summary_statistics']
        print(f"Average turnover ratio: {summary['avg_turnover_ratio']:.2f}")
        print(f"Average days of inventory: {summary['avg_days_of_inventory']:.1f}")
        print(f"Fast movers: {summary['fast_movers']}")
        print(f"Slow movers: {summary['slow_movers']}")
        
        print("\nTurnover metrics:")
        metrics_df = turnover_result['turnover_metrics']
        for _, metric in metrics_df.head().iterrows():
            print(f"  {metric['sku']}: Turnover = {metric['turnover_ratio']:.2f}, "
                  f"Days = {metric['days_of_inventory']:.0f}, Class = {metric['turnover_class']}")
    
    # Example 5: Working with different data formats
    print("\n5. Data Format Integration Example")
    print("-" * 40)
    
    # CSV integration
    csv_data = inventory_service.to_csv(inventory_df)
    print(f"Exported to CSV ({len(csv_data)} characters)")
    
    imported_df = inventory_service.from_csv(csv_data)
    print(f"Imported from CSV: {len(imported_df)} rows")
    
    # JSON integration
    json_data = inventory_service.to_json(inventory_df.head(2))
    print(f"Exported to JSON: {len(json_data)} records")
    
    json_df = inventory_service.from_json(json_data)
    print(f"Imported from JSON: {len(json_df)} rows")
    
    print("\n=== Inventory Domain Example Complete ===")
    print("Key Benefits:")
    print("- Statistical safety stock calculations")
    print("- ABC classification for prioritization")
    print("- Real-time stock alerts")
    print("- Turnover analysis for optimization")
    print("- Multiple data format support")


if __name__ == "__main__":
    main()