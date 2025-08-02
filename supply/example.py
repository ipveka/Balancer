"""
Supply Domain Example Usage

This example demonstrates how to use the supply domain for procurement
and manufacturing optimization using the new service layer with DataFrames.
"""

import pandas as pd
from supply.service import SupplyService
from utils.dummy_data import DummyDataGenerator


def main():
    print("=== Supply Domain Example ===")
    
    # Initialize service
    supply_service = SupplyService()
    
    # Generate sample data
    generator = DummyDataGenerator(seed=42)
    
    # Example 1: Procurement Optimization with DataFrame
    print("\n1. Procurement Optimization Example")
    print("-" * 40)
    
    # Generate sample procurement data
    procurement_data = generator.generate_procurement_data(5)
    procurement_df = pd.DataFrame(procurement_data)
    
    print("Input data:")
    print(procurement_df[['sku', 'current_inventory', 'forecast_demand_4weeks', 'unit_cost']].head())
    
    # Optimize procurement
    result = supply_service.optimize_procurement(procurement_df)
    
    if result['success']:
        print(f"\nOptimization successful! Generated {result['recommendations_count']} recommendations")
        print(f"Total cost: ${result['optimization_summary']['total_cost']:,.2f}")
        print(f"Processing time: {result['processing_time_seconds']} seconds")
        
        print("\nRecommendations:")
        recommendations_df = result['recommendations']
        for _, rec in recommendations_df.head(3).iterrows():
            print(f"  {rec['sku']}: Order {rec['recommended_quantity']} units from {rec['supplier_id']} "
                  f"(${rec['total_cost']:.2f}) - {rec['recommendation_action']}")
    
    # Example 2: Manufacturing Optimization
    print("\n2. Manufacturing Optimization Example")
    print("-" * 40)
    
    # Generate sample manufacturing data
    manufacturing_data = generator.generate_manufacturing_data(5)
    manufacturing_df = pd.DataFrame(manufacturing_data)
    
    print("Input data:")
    print(manufacturing_df[['sku', 'current_inventory', 'batch_size', 'production_time_days']].head())
    
    # Optimize manufacturing
    result = supply_service.optimize_manufacturing(manufacturing_df)
    
    if result['success']:
        print(f"\nOptimization successful! Generated {result['recommendations_count']} recommendations")
        print(f"Total cost: ${result['optimization_summary']['total_cost']:,.2f}")
        
        print("\nRecommendations:")
        recommendations_df = result['recommendations']
        for _, rec in recommendations_df.head(3).iterrows():
            print(f"  {rec['sku']}: Produce {rec['recommended_batch_qty']} units "
                  f"(Start: {rec['production_start_date']}) - {rec['recommendation_action']}")
    
    # Example 3: Supplier Comparison
    print("\n3. Supplier Comparison Example")
    print("-" * 40)
    
    # Create sample supplier comparison data
    supplier_data = [
        {'sku': 'SKU-001', 'supplier_id': 'SUP-A', 'unit_cost': 20.00, 'lead_time_days': 7, 'quality_rating': 4.5, 'min_order_qty': 100},
        {'sku': 'SKU-001', 'supplier_id': 'SUP-B', 'unit_cost': 18.50, 'lead_time_days': 10, 'quality_rating': 4.2, 'min_order_qty': 150},
        {'sku': 'SKU-001', 'supplier_id': 'SUP-C', 'unit_cost': 22.00, 'lead_time_days': 5, 'quality_rating': 4.8, 'min_order_qty': 200},
        {'sku': 'SKU-002', 'supplier_id': 'SUP-A', 'unit_cost': 15.00, 'lead_time_days': 7, 'quality_rating': 4.5, 'min_order_qty': 200},
        {'sku': 'SKU-002', 'supplier_id': 'SUP-D', 'unit_cost': 14.25, 'lead_time_days': 12, 'quality_rating': 4.0, 'min_order_qty': 250}
    ]
    supplier_df = pd.DataFrame(supplier_data)
    
    print("Supplier comparison data:")
    print(supplier_df[['sku', 'supplier_id', 'unit_cost', 'lead_time_days', 'quality_rating']].head())
    
    # Compare suppliers
    comparison_result = supply_service.compare_suppliers(supplier_df)
    
    if comparison_result['success']:
        print(f"\nComparison complete! Analyzed {comparison_result['skus_analyzed']} SKUs")
        
        print("\nRecommended suppliers:")
        recommendations = comparison_result['recommendations']
        for _, rec in recommendations.iterrows():
            print(f"  {rec['sku']}: {rec['supplier_id']} (Score: {rec['composite_score']:.3f}, "
                  f"Cost: ${rec['unit_cost']:.2f}, Quality: {rec['quality_rating']}/5)")
    
    # Example 4: Working with CSV data
    print("\n4. CSV Integration Example")
    print("-" * 40)
    
    # Convert DataFrame to CSV
    csv_data = supply_service.to_csv(procurement_df)
    print(f"Exported to CSV ({len(csv_data)} characters)")
    
    # Convert CSV back to DataFrame
    imported_df = supply_service.from_csv(csv_data)
    print(f"Imported from CSV: {len(imported_df)} rows, {len(imported_df.columns)} columns")
    
    # Example 5: Working with JSON data (for API integration)
    print("\n5. JSON Integration Example")
    print("-" * 40)
    
    # Convert DataFrame to JSON
    json_data = supply_service.to_json(procurement_df.head(2))
    print(f"Exported to JSON: {len(json_data)} records")
    print(f"First record keys: {list(json_data[0].keys())}")
    
    # Convert JSON back to DataFrame
    json_df = supply_service.from_json(json_data)
    print(f"Imported from JSON: {len(json_df)} rows, {len(json_df.columns)} columns")
    
    print("\n=== Supply Domain Example Complete ===")
    print("Key Benefits:")
    print("- Works with DataFrames as primary format")
    print("- Supports CSV and JSON conversion")
    print("- Comprehensive optimization algorithms")
    print("- Detailed performance metrics")
    print("- Easy integration with APIs and data pipelines")


if __name__ == "__main__":
    main()