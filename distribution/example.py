"""
Distribution Domain Example Usage

This example demonstrates how to use the distribution domain for vehicle routing
and logistics optimization using the new service layer with DataFrames.
"""

import pandas as pd
from distribution.service import DistributionService
from utils.dummy_data import DummyDataGenerator


def main():
    print("=== Distribution Domain Example ===")
    
    # Initialize service
    distribution_service = DistributionService()
    
    # Generate sample data
    generator = DummyDataGenerator(seed=42)
    
    # Example 1: Basic Route Optimization
    print("\n1. Basic Route Optimization Example")
    print("-" * 40)
    
    # Generate sample orders and vehicles data
    distribution_data = generator.generate_orders_and_vehicles(8, 3)
    orders_df = pd.DataFrame(distribution_data['orders'])
    vehicles_df = pd.DataFrame(distribution_data['vehicles'])
    
    print("Orders data:")
    print(orders_df[['order_id', 'customer_lat', 'customer_lon', 'volume_m3', 'weight_kg']].head())
    
    print("\nVehicles data:")
    print(vehicles_df[['vehicle_id', 'max_volume_m3', 'max_weight_kg', 'cost_per_km']])
    
    # Optimize routes
    optimization_params = {
        'algorithm_preference': 'greedy',
        'max_route_distance_km': 150.0,
        'depot_location': {'lat': 40.7128, 'lon': -74.0060}  # NYC
    }
    
    result = distribution_service.optimize_routes(orders_df, vehicles_df, optimization_params)
    
    if result['success']:
        solution = result['solution']
        print(f"\nOptimization successful!")
        print(f"Total distance: {solution['total_distance']:.1f} km")
        print(f"Total cost: ${solution['total_cost']:.2f}")
        print(f"Number of routes: {solution['num_routes']}")
        print(f"Orders assigned: {solution['num_orders_assigned']}/{len(orders_df)}")
        print(f"Algorithm used: {solution['algorithm_used']}")
        print(f"Processing time: {solution['optimization_time_seconds']:.3f} seconds")
        
        print("\nRoute assignments:")
        assignments_df = result['assignments']
        for vehicle_id in assignments_df['vehicle_id'].unique():
            vehicle_assignments = assignments_df[assignments_df['vehicle_id'] == vehicle_id]
            total_cost = vehicle_assignments['total_cost'].sum()
            print(f"  {vehicle_id}: {len(vehicle_assignments)} orders, ${total_cost:.2f} total cost")
            
            for _, assignment in vehicle_assignments.head(3).iterrows():
                print(f"    {assignment['sequence']}. {assignment['order_id']} "
                      f"({assignment['distance_km']:.1f} km, ${assignment['total_cost']:.2f})")
        
        print("\nVehicle utilization:")
        for vehicle_id, utilization in result['vehicle_utilization'].items():
            print(f"  {vehicle_id}: Volume = {utilization['volume_utilization_percent']:.1f}%, "
                  f"Weight = {utilization['weight_utilization_percent']:.1f}%")
    
    # Example 2: Distance Matrix Calculation
    print("\n2. Distance Matrix Calculation Example")
    print("-" * 40)
    
    # Create sample locations
    locations_data = [
        {'location_id': 'DEPOT', 'latitude': 40.7128, 'longitude': -74.0060},
        {'location_id': 'CUST-001', 'latitude': 40.7589, 'longitude': -73.9851},
        {'location_id': 'CUST-002', 'latitude': 40.6892, 'longitude': -74.0445},
        {'location_id': 'CUST-003', 'latitude': 40.7505, 'longitude': -73.9934}
    ]
    locations_df = pd.DataFrame(locations_data)
    
    print("Locations:")
    print(locations_df)
    
    # Calculate distance matrix
    distance_result = distribution_service.calculate_distance_matrix(
        locations_df, 
        {'calculation_method': 'haversine'}
    )
    
    if distance_result['success']:
        print(f"\nDistance matrix calculated!")
        print(f"Method: {distance_result['calculation_method']}")
        print(f"Locations: {distance_result['locations_count']}")
        
        print("\nDistance matrix (km):")
        distances = distance_result['distance_matrix']
        location_ids = list(distances.keys())
        
        # Print header
        print(f"{'':>10}", end="")
        for loc_id in location_ids:
            print(f"{loc_id:>10}", end="")
        print()
        
        # Print matrix
        for from_id in location_ids:
            print(f"{from_id:>10}", end="")
            for to_id in location_ids:
                print(f"{distances[from_id][to_id]:>10.1f}", end="")
            print()
    
    # Example 3: Route Performance Analysis
    print("\n3. Route Performance Analysis Example")
    print("-" * 40)
    
    if result['success']:
        routes_df = result['routes']
        
        print("Route performance data:")
        print(routes_df[['route_id', 'vehicle_id', 'total_distance_km', 'total_cost', 'estimated_duration_hours']])
        
        # Analyze performance
        performance_result = distribution_service.analyze_route_performance(routes_df)
        
        if performance_result['success']:
            metrics = performance_result['performance_metrics']
            print(f"\nPerformance Analysis:")
            print(f"Total routes: {metrics['total_routes']}")
            print(f"Average route distance: {metrics['avg_route_distance']:.1f} km")
            print(f"Average route cost: ${metrics['avg_route_cost']:.2f}")
            print(f"Cost per km: ${metrics['cost_per_km']:.2f}")
            print(f"Longest route: {metrics['longest_route_distance']:.1f} km")
            print(f"Shortest route: {metrics['shortest_route_distance']:.1f} km")
            
            # Show recommendations
            recommendations = performance_result['recommendations']
            if recommendations:
                print("\nRecommendations:")
                for rec in recommendations:
                    print(f"  {rec['category']}: {rec['recommendation']}")
    
    # Example 4: Capacity Utilization Analysis
    print("\n4. Capacity Utilization Analysis Example")
    print("-" * 40)
    
    if result['success']:
        assignments_df = result['assignments']
        
        # Analyze capacity utilization
        utilization_result = distribution_service.analyze_capacity_utilization(
            assignments_df, vehicles_df, orders_df
        )
        
        if utilization_result['success']:
            print("Capacity utilization analysis:")
            
            utilization_df = utilization_result['vehicle_utilization']
            for _, util in utilization_df.iterrows():
                print(f"\n  {util['vehicle_id']}:")
                print(f"    Volume: {util['total_volume_used_m3']:.1f}/{util['max_volume_m3']:.1f} m³ "
                      f"({util['volume_utilization_percent']:.1f}%)")
                print(f"    Weight: {util['total_weight_used_kg']:.1f}/{util['max_weight_kg']:.1f} kg "
                      f"({util['weight_utilization_percent']:.1f}%)")
                print(f"    Orders: {util['orders_assigned']}")
                print(f"    Overloaded: {'Yes' if util['is_overloaded'] else 'No'}")
            
            summary = utilization_result['summary_statistics']
            print(f"\nSummary:")
            print(f"Average utilization: {summary['avg_overall_utilization']:.1f}%")
            print(f"Overloaded vehicles: {summary['overloaded_vehicles']}")
            print(f"Underutilized vehicles: {summary['underutilized_vehicles']}")
            print(f"Well-utilized vehicles: {summary['well_utilized_vehicles']}")
            
            # Show recommendations
            recommendations = utilization_result['recommendations']
            if recommendations:
                print("\nUtilization recommendations:")
                for rec in recommendations:
                    print(f"  {rec['category']}: {rec['recommendation']}")
    
    # Example 5: Algorithm Comparison
    print("\n5. Algorithm Comparison Example")
    print("-" * 40)
    
    # Test different algorithms
    algorithms = ['greedy', 'nearest_neighbor']
    algorithm_results = {}
    
    for algorithm in algorithms:
        params = optimization_params.copy()
        params['algorithm_preference'] = algorithm
        
        alg_result = distribution_service.optimize_routes(orders_df, vehicles_df, params)
        
        if alg_result['success']:
            solution = alg_result['solution']
            algorithm_results[algorithm] = {
                'total_distance': solution['total_distance'],
                'total_cost': solution['total_cost'],
                'processing_time': solution['optimization_time_seconds'],
                'orders_assigned': solution['num_orders_assigned']
            }
    
    print("Algorithm comparison:")
    for algorithm, metrics in algorithm_results.items():
        print(f"\n  {algorithm.title()}:")
        print(f"    Distance: {metrics['total_distance']:.1f} km")
        print(f"    Cost: ${metrics['total_cost']:.2f}")
        print(f"    Time: {metrics['processing_time']:.3f} seconds")
        print(f"    Orders assigned: {metrics['orders_assigned']}")
    
    # Find best algorithm
    if len(algorithm_results) > 1:
        best_algorithm = min(algorithm_results.keys(), 
                           key=lambda x: algorithm_results[x]['total_cost'])
        print(f"\nBest algorithm by cost: {best_algorithm.title()}")
    
    # Example 6: Data Format Integration
    print("\n6. Data Format Integration Example")
    print("-" * 40)
    
    # CSV integration
    orders_csv = distribution_service.to_csv(orders_df)
    print(f"Orders exported to CSV ({len(orders_csv)} characters)")
    
    imported_orders = distribution_service.from_csv(orders_csv)
    print(f"Orders imported from CSV: {len(imported_orders)} rows")
    
    # JSON integration
    vehicles_json = distribution_service.to_json(vehicles_df)
    print(f"Vehicles exported to JSON: {len(vehicles_json)} records")
    
    imported_vehicles = distribution_service.from_json(vehicles_json)
    print(f"Vehicles imported from JSON: {len(imported_vehicles)} rows")
    
    print("\n=== Distribution Domain Example Complete ===")
    print("Key Benefits:")
    print("- Advanced VRP optimization algorithms")
    print("- Comprehensive route performance analysis")
    print("- Capacity utilization optimization")
    print("- Distance matrix calculations")
    print("- Multiple algorithm support")
    print("- Real-time optimization capabilities")
    print("- Detailed analytics and recommendations")


if __name__ == "__main__":
    main()