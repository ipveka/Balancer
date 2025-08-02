"""
Demand Domain Example Usage

This example demonstrates how to use the demand domain for forecasting
and demand pattern analysis using the new service layer with DataFrames.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from demand.service import DemandService
from utils.dummy_data import DummyDataGenerator


def main():
    print("=== Demand Domain Example ===")
    
    # Initialize service
    demand_service = DemandService()
    
    # Generate sample data
    generator = DummyDataGenerator(seed=42)
    
    # Example 1: Demand Forecasting
    print("\n1. Demand Forecasting Example")
    print("-" * 40)
    
    # Generate sample demand data (time series)
    demand_data = generator.generate_demand_data(100, 3)  # 100 records, 3 SKUs
    demand_df = pd.DataFrame(demand_data)
    
    print("Input data sample:")
    print(demand_df.head())
    print(f"Total records: {len(demand_df)}, SKUs: {len(demand_df['sku'].unique())}")
    print(f"Date range: {demand_df['date'].min()} to {demand_df['date'].max()}")
    
    # Generate forecast
    forecast_params = {
        'forecast_horizon_weeks': 8,
        'confidence_level': 0.95,
        'include_seasonality': True
    }
    
    result = demand_service.generate_forecast(demand_df, forecast_params)
    
    if result['success']:
        print(f"\nForecast successful! Generated {result['forecasts_count']} predictions")
        print(f"Processing time: {result['processing_time_seconds']} seconds")
        
        forecast_summary = result['forecast_summary']
        print(f"Total forecasted demand: {forecast_summary['total_forecasted_demand']:,.0f}")
        print(f"Trend direction: {forecast_summary['trend_direction']}")
        print(f"Confidence score: {forecast_summary['confidence_score']:.3f}")
        
        print("\nSample forecasts:")
        forecasts_df = result['forecasts']
        for sku in forecasts_df['sku'].unique()[:2]:  # Show first 2 SKUs
            sku_forecasts = forecasts_df[forecasts_df['sku'] == sku].head(3)
            print(f"\n  {sku}:")
            for _, forecast in sku_forecasts.iterrows():
                print(f"    {forecast['forecast_date']}: {forecast['prediction']:.1f} "
                      f"({forecast['confidence_interval_lower']:.1f} - {forecast['confidence_interval_upper']:.1f})")
    
    # Example 2: Demand Pattern Analysis
    print("\n2. Demand Pattern Analysis Example")
    print("-" * 40)
    
    # Analyze demand patterns
    analysis_result = demand_service.analyze_demand_patterns(demand_df)
    
    if analysis_result['success']:
        print(f"Pattern analysis complete! Analyzed {analysis_result['analyses_count']} SKUs")
        
        summary = analysis_result['summary_statistics']
        print(f"Average coefficient of variation: {summary['avg_coefficient_variation']:.3f}")
        print(f"Seasonal items: {summary['seasonal_items_count']}")
        print(f"High variability items: {summary['high_variability_items']}")
        print(f"Trending up items: {summary['trending_up_items']}")
        
        print("\nPattern analysis by SKU:")
        analyses_df = analysis_result['analyses']
        for _, analysis in analyses_df.head(3).iterrows():
            print(f"  {analysis['sku']}:")
            print(f"    Trend: {analysis['trend_direction']} (coeff: {analysis['trend_coefficient']:.4f})")
            print(f"    Seasonality: {analysis['seasonality_pattern']}")
            print(f"    Volatility: {analysis['volatility_level']}")
            print(f"    Data quality: {analysis['data_quality_score']:.3f}")
    
    # Example 3: Forecast Accuracy Validation
    print("\n3. Forecast Accuracy Validation Example")
    print("-" * 40)
    
    # Create sample historical vs forecast data for validation
    # Split demand data into "historical" and "forecast" for demonstration
    split_date = demand_df['date'].quantile(0.8)  # Use 80% as historical, 20% as "forecast"
    historical_df = demand_df[demand_df['date'] <= split_date].copy()
    validation_df = demand_df[demand_df['date'] > split_date].copy()
    
    # Create mock forecast data (in real scenario, this would be actual forecasts)
    forecast_validation_df = validation_df.copy()
    forecast_validation_df['prediction'] = forecast_validation_df['quantity'] * np.random.uniform(0.9, 1.1, len(forecast_validation_df))
    forecast_validation_df = forecast_validation_df.rename(columns={'quantity': 'actual_quantity'})
    
    print(f"Historical data: {len(historical_df)} records")
    print(f"Validation data: {len(validation_df)} records")
    
    # Validate forecast accuracy
    validation_result = demand_service.validate_forecast_accuracy(
        validation_df, 
        forecast_validation_df[['date', 'sku', 'prediction']]
    )
    
    if validation_result['success']:
        print(f"\nValidation complete!")
        
        summary = validation_result['validation_summary']
        print(f"SKUs evaluated: {summary['skus_evaluated']}")
        print(f"Total forecasts: {summary['total_forecasts']}")
        print(f"Average accuracy: {summary['avg_accuracy_percentage']:.1f}%")
        
        overall = validation_result['overall_metrics']
        print(f"Overall MAPE: {overall['overall_mape']:.2f}%")
        print(f"Overall directional accuracy: {overall['overall_directional_accuracy']:.1f}%")
        
        print("\nAccuracy by SKU:")
        accuracy_df = validation_result['accuracy_results']
        for _, acc in accuracy_df.head(3).iterrows():
            print(f"  {acc['sku']}: MAPE = {acc['mean_absolute_percentage_error']:.2f}%, "
                  f"Accuracy = {acc['forecast_accuracy_percentage']:.1f}%")
    
    # Example 4: Advanced Demand Analysis
    print("\n4. Advanced Demand Analysis Example")
    print("-" * 40)
    
    # Create more detailed analysis
    print("Demand statistics by SKU:")
    for sku in demand_df['sku'].unique():
        sku_data = demand_df[demand_df['sku'] == sku]
        
        total_demand = sku_data['quantity'].sum()
        avg_demand = sku_data['quantity'].mean()
        std_demand = sku_data['quantity'].std()
        cv = std_demand / avg_demand if avg_demand > 0 else 0
        
        print(f"  {sku}:")
        print(f"    Total demand: {total_demand:,}")
        print(f"    Average: {avg_demand:.1f} ± {std_demand:.1f}")
        print(f"    Coefficient of variation: {cv:.3f}")
        
        # Trend analysis
        sku_data_sorted = sku_data.sort_values('date')
        if len(sku_data_sorted) > 1:
            trend = np.polyfit(range(len(sku_data_sorted)), sku_data_sorted['quantity'], 1)[0]
            trend_direction = "↗" if trend > 0.1 else "↘" if trend < -0.1 else "→"
            print(f"    Trend: {trend_direction} ({trend:.2f} units/period)")
    
    # Example 5: Data Format Integration
    print("\n5. Data Format Integration Example")
    print("-" * 40)
    
    # CSV integration
    csv_data = demand_service.to_csv(demand_df.head(10))
    print(f"Exported to CSV ({len(csv_data)} characters)")
    
    imported_df = demand_service.from_csv(csv_data)
    print(f"Imported from CSV: {len(imported_df)} rows")
    
    # JSON integration
    json_data = demand_service.to_json(demand_df.head(5))
    print(f"Exported to JSON: {len(json_data)} records")
    
    json_df = demand_service.from_json(json_data)
    print(f"Imported from JSON: {len(json_df)} rows")
    
    # Example 6: Custom Forecast Parameters
    print("\n6. Custom Forecast Parameters Example")
    print("-" * 40)
    
    # Test different forecast parameters
    custom_params = {
        'forecast_horizon_weeks': 4,
        'confidence_level': 0.90,
        'include_seasonality': False,
        'algorithm': 'simple_trend'
    }
    
    custom_result = demand_service.generate_forecast(demand_df, custom_params)
    
    if custom_result['success']:
        print(f"Custom forecast generated: {custom_result['forecasts_count']} predictions")
        print(f"Parameters used: {custom_result['parameters_used']}")
        
        # Compare with original forecast
        original_total = result['forecast_summary']['total_forecasted_demand']
        custom_total = custom_result['forecast_summary']['total_forecasted_demand']
        difference = ((custom_total - original_total) / original_total) * 100
        
        print(f"Forecast difference: {difference:+.1f}% vs standard parameters")
    
    print("\n=== Demand Domain Example Complete ===")
    print("Key Benefits:")
    print("- ML-based demand forecasting")
    print("- Comprehensive pattern analysis")
    print("- Forecast accuracy validation")
    print("- Trend and seasonality detection")
    print("- Flexible parameter configuration")
    print("- Multiple data format support")


if __name__ == "__main__":
    main()