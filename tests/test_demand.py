"""
Demand Module Tests

Comprehensive test suite for the demand forecasting module including
API endpoint tests, service logic tests, LightGBM model validation,
and error handling for AI-powered demand forecasting.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json
import io

from demand.api import router as demand_router
from demand.service import DemandService, process_demand_csv_file, analyze_demand_data
from demand.models import (
    DemandForecastRequest, DemandAnalyticsRequest,
    DemandDataInput, ForecastOutput, ForecastAccuracy,
    TrendDirection, SeasonalityPattern,
    DemandException, InvalidForecastException, InsufficientDataException,
    ModelTrainingException, ForecastGenerationException, DataQualityException
)
from utils.helpers import CSVProcessingError, DataValidationError


# Test fixtures
@pytest.fixture
def app():
    """Create FastAPI app with demand router for testing."""
    app = FastAPI()
    app.include_router(demand_router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def demand_service():
    """Create demand service instance."""
    return DemandService()


@pytest.fixture
def sample_demand_csv():
    """Sample demand CSV data for testing (sufficient for ML training)."""
    csv_lines = ["date,sku,quantity"]
    
    # Generate 8 weeks of daily data for 2 SKUs
    base_date = datetime.now() - timedelta(weeks=8)
    
    for day in range(56):  # 8 weeks * 7 days
        current_date = base_date + timedelta(days=day)
        date_str = current_date.strftime('%Y-%m-%d')
        
        # SKU 1: Stable demand with small variation
        demand1 = 100 + (day % 7 - 3) * 5 + int((hash(date_str + "SKU1") % 21) - 10)
        demand1 = max(0, demand1)
        csv_lines.append(f"{date_str},WIDGET-001,{demand1}")
        
        # SKU 2: Trending demand
        trend_factor = 1 + (day / 56) * 0.2  # 20% growth
        demand2 = int(80 * trend_factor) + int((hash(date_str + "SKU2") % 15) - 7)
        demand2 = max(0, demand2)
        csv_lines.append(f"{date_str},GADGET-002,{demand2}")
    
    return '\n'.join(csv_lines)


@pytest.fixture
def insufficient_data_csv():
    """CSV with insufficient data for ML training."""
    csv_lines = ["date,sku,quantity"]
    
    # Only 2 weeks of data (insufficient)
    base_date = datetime.now() - timedelta(weeks=2)
    
    for day in range(14):
        current_date = base_date + timedelta(days=day)
        date_str = current_date.strftime('%Y-%m-%d')
        demand = 50 + day
        csv_lines.append(f"{date_str},SHORT-001,{demand}")
    
    return '\n'.join(csv_lines)


@pytest.fixture
def invalid_csv():
    """Invalid CSV data for error testing."""
    return """invalid,headers,format
some,data,here"""


@pytest.fixture
def missing_columns_csv():
    """CSV with missing required columns."""
    return """date,sku
2024-01-01,WIDGET-001"""


@pytest.fixture
def invalid_date_csv():
    """CSV with invalid date format."""
    return """date,sku,quantity
invalid-date,WIDGET-001,100
2024-01-02,WIDGET-001,110"""


# API Endpoint Tests

class TestDemandAPI:
    """Test class for demand API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/demand/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["module"] == "demand"
        assert "supported_operations" in data
        assert "forecasting" in data["supported_operations"]
        assert "analytics" in data["supported_operations"]
        assert data["ml_model"] == "LightGBM"
    
    def test_get_forecast_horizons(self, client):
        """Test get forecast horizons endpoint."""
        response = client.get("/demand/horizons")
        assert response.status_code == 200
        
        data = response.json()
        assert "supported_horizons" in data
        assert len(data["supported_horizons"]) >= 1
        assert "default_horizon_weeks" in data
        assert "confidence_levels" in data
    
    def test_get_accuracy_levels(self, client):
        """Test get accuracy levels endpoint."""
        response = client.get("/demand/accuracy-levels")
        assert response.status_code == 200
        
        data = response.json()
        assert "accuracy_levels" in data
        assert len(data["accuracy_levels"]) == 3
        
        levels = [level["level"] for level in data["accuracy_levels"]]
        assert "HIGH" in levels
        assert "MEDIUM" in levels
        assert "LOW" in levels
    
    def test_get_pattern_types(self, client):
        """Test get pattern types endpoint."""
        response = client.get("/demand/patterns")
        assert response.status_code == 200
        
        data = response.json()
        assert "trend_patterns" in data
        assert "seasonality_patterns" in data
        assert len(data["trend_patterns"]) == 4
        assert len(data["seasonality_patterns"]) == 5
    
    def test_get_demand_status(self, client):
        """Test get demand status endpoint."""
        response = client.get("/demand/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["module"] == "demand"
        assert data["status"] == "operational"
        assert "features" in data
        assert "ml_configuration" in data
        assert data["features"]["ai_forecasting"] is True
        assert data["features"]["lightgbm_model"] is True
        assert data["ml_configuration"]["model_type"] == "LightGBM"
    
    def test_download_demand_template(self, client):
        """Test demand template download."""
        response = client.get("/demand/template")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "demand_template.csv" in response.headers["content-disposition"]
        
        # Verify CSV content
        content = response.content.decode('utf-8')
        assert "date,sku,quantity" in content
        assert "WIDGET-001" in content
    
    def test_generate_demand_forecast(self, client, sample_demand_csv):
        """Test demand forecast generation endpoint."""
        request_data = {
            "csv_data": sample_demand_csv,
            "forecast_horizon_weeks": 4,
            "confidence_level": 0.95,
            "include_seasonality": True,
            "model_params": {}
        }
        
        response = client.post("/demand/forecast", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["forecasts_count"] > 0
        assert "csv_output" in data
        assert "processing_time_seconds" in data
        assert "model_performance" in data
        assert "forecast_summary" in data
    
    def test_generate_forecast_invalid_csv(self, client, invalid_csv):
        """Test forecast generation with invalid CSV data."""
        request_data = {
            "csv_data": invalid_csv,
            "forecast_horizon_weeks": 4,
            "confidence_level": 0.95,
            "include_seasonality": True,
            "model_params": {}
        }
        
        response = client.post("/demand/forecast", json=request_data)
        assert response.status_code == 400
        assert "CSV processing failed" in response.json()["detail"]
    
    def test_generate_forecast_insufficient_data(self, client, insufficient_data_csv):
        """Test forecast generation with insufficient data."""
        request_data = {
            "csv_data": insufficient_data_csv,
            "forecast_horizon_weeks": 4,
            "confidence_level": 0.95,
            "include_seasonality": True,
            "model_params": {}
        }
        
        response = client.post("/demand/forecast", json=request_data)
        assert response.status_code == 400
        assert "Insufficient data" in response.json()["detail"]
    
    def test_generate_forecast_missing_columns(self, client, missing_columns_csv):
        """Test forecast generation with missing required columns."""
        request_data = {
            "csv_data": missing_columns_csv,
            "forecast_horizon_weeks": 4,
            "confidence_level": 0.95,
            "include_seasonality": True,
            "model_params": {}
        }
        
        response = client.post("/demand/forecast", json=request_data)
        assert response.status_code == 400
        assert "Missing required columns" in response.json()["detail"]
    
    def test_upload_demand_csv(self, client, sample_demand_csv):
        """Test demand CSV file upload."""
        files = {"file": ("test.csv", io.StringIO(sample_demand_csv), "text/csv")}
        data = {"forecast_horizon_weeks": 6, "confidence_level": 0.95}
        
        response = client.post("/demand/upload", files=files, data=data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "demand_forecasts.csv" in response.headers["content-disposition"]
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        files = {"file": ("test.txt", io.StringIO("test content"), "text/plain")}
        
        response = client.post("/demand/upload", files=files)
        assert response.status_code == 400
        assert "File must be a CSV file" in response.json()["detail"]
    
    def test_analyze_demand_patterns(self, client):
        """Test demand pattern analysis endpoint."""
        request_data = {
            "sku": "WIDGET-001",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "include_outliers": False,
            "seasonality_analysis": True
        }
        
        response = client.post("/demand/analytics", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "analytics_count" in data
        assert "processing_time_seconds" in data
        assert "summary_statistics" in data
    
    def test_get_model_info_not_found(self, client):
        """Test get model info for non-existent SKU."""
        response = client.get("/demand/models/NONEXISTENT-001")
        assert response.status_code == 404
        assert "No trained model found" in response.json()["detail"]
    
    def test_get_all_models_empty(self, client):
        """Test get all models when none are trained."""
        response = client.get("/demand/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_models"] == 0
        assert "models" in data
        assert "last_updated" in data


# Service Layer Tests

class TestDemandService:
    """Test class for demand service layer."""
    
    @pytest.mark.asyncio
    async def test_process_demand_csv_success(self, demand_service, sample_demand_csv):
        """Test successful demand CSV processing and forecasting."""
        result = await demand_service.process_demand_csv(
            sample_demand_csv, 
            forecast_horizon_weeks=4,
            confidence_level=0.95
        )
        
        assert result.total_forecasts > 0
        assert len(result.forecasts) > 0
        
        # Check first forecast
        forecast = result.forecasts[0]
        assert forecast.sku in ["WIDGET-001", "GADGET-002"]
        assert forecast.prediction >= 0
        assert forecast.confidence_interval_lower >= 0
        assert forecast.confidence_interval_upper >= forecast.confidence_interval_lower
        assert forecast.confidence_level == 0.95
        assert forecast.forecast_accuracy in [acc.value for acc in ForecastAccuracy]
        assert forecast.model_version == demand_service.model_version
    
    @pytest.mark.asyncio
    async def test_process_demand_csv_insufficient_data(self, demand_service, insufficient_data_csv):
        """Test demand CSV processing with insufficient data."""
        with pytest.raises(InsufficientDataException):
            await demand_service.process_demand_csv(insufficient_data_csv)
    
    @pytest.mark.asyncio
    async def test_process_demand_csv_invalid_data(self, demand_service, invalid_csv):
        """Test demand CSV processing with invalid data."""
        with pytest.raises(CSVProcessingError):
            await demand_service.process_demand_csv(invalid_csv)
    
    @pytest.mark.asyncio
    async def test_generate_sku_forecasts(self, demand_service, sample_demand_csv):
        """Test SKU-specific forecast generation."""
        # First process the CSV to get DataFrame
        from utils.helpers import read_csv_to_dict, validate_pydantic_data
        import pandas as pd
        
        raw_data = read_csv_to_dict(sample_demand_csv, ['date', 'sku', 'quantity'])
        processed_data = []
        for row in raw_data:
            processed_data.append({
                'date': row['date'].strip(),
                'sku': row['sku'].strip().upper(),
                'quantity': int(row['quantity'])
            })
        
        validated_data = validate_pydantic_data(processed_data, DemandDataInput)
        df = pd.DataFrame([{
            'date': item.date,
            'sku': item.sku,
            'quantity': item.quantity
        } for item in validated_data])
        
        # Test forecast generation for specific SKU
        sku = "WIDGET-001"
        forecasts, performance = await demand_service._generate_sku_forecasts(
            df, sku, forecast_horizon_weeks=4, confidence_level=0.95
        )
        
        assert len(forecasts) == 28  # 4 weeks * 7 days
        assert all(f.sku == sku for f in forecasts)
        assert all(f.prediction >= 0 for f in forecasts)
        
        # Check performance metrics
        assert 'train_mae' in performance
        assert 'val_mae' in performance
        assert 'val_mape' in performance
        assert performance['val_mae'] >= 0
        assert performance['val_mape'] >= 0
    
    @pytest.mark.asyncio
    async def test_prepare_features(self, demand_service):
        """Test feature engineering for ML model."""
        import pandas as pd
        
        # Create sample SKU data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        sku_data = pd.DataFrame({
            'date': dates,
            'sku': 'TEST-001',
            'quantity': [100 + i + (i % 7) * 10 for i in range(30)]
        })
        
        features_df = await demand_service._prepare_features(sku_data)
        
        # Check that features were created
        expected_features = [
            'quantity', 'date',
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_7', 'lag_14',
            'ma_3', 'ma_7', 'ma_14', 'ma_28',
            'year', 'month', 'quarter', 'week_of_year', 'day_of_week',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            'trend', 'rolling_mean_7', 'rolling_std_7'
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns, f"Missing feature: {feature}"
        
        # Check that no NaN values remain
        assert not features_df.isnull().any().any(), "Features contain NaN values"
    
    @pytest.mark.asyncio
    async def test_train_lightgbm_model(self, demand_service):
        """Test LightGBM model training."""
        import pandas as pd
        
        # Create sample features data
        n_samples = 50
        features_df = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=n_samples, freq='D'),
            'quantity': [100 + i + (i % 7) * 5 for i in range(n_samples)],
            'lag_1': [95 + i + ((i-1) % 7) * 5 for i in range(n_samples)],
            'lag_7': [100 + max(0, i-7) + ((i-7) % 7) * 5 for i in range(n_samples)],
            'ma_7': [100 + i for i in range(n_samples)],
            'trend': list(range(n_samples)),
            'month': [1] * n_samples,
            'day_of_week': [i % 7 for i in range(n_samples)]
        })
        
        model, performance = await demand_service._train_lightgbm_model(features_df, "TEST-001")
        
        # Check model was trained
        assert model is not None
        assert "TEST-001" in demand_service.trained_models
        assert "TEST-001" in demand_service.model_metadata
        
        # Check performance metrics
        assert 'train_mae' in performance
        assert 'val_mae' in performance
        assert 'train_rmse' in performance
        assert 'val_rmse' in performance
        assert 'train_mape' in performance
        assert 'val_mape' in performance
        assert 'feature_importance' in performance
        
        # Check all metrics are non-negative
        for metric in ['train_mae', 'val_mae', 'train_rmse', 'val_rmse', 'train_mape', 'val_mape']:
            assert performance[metric] >= 0
    
    @pytest.mark.asyncio
    async def test_export_forecast_csv(self, demand_service, sample_demand_csv):
        """Test forecast CSV export."""
        forecasts = await demand_service.process_demand_csv(sample_demand_csv, forecast_horizon_weeks=2)
        csv_output = await demand_service.export_forecast_csv(forecasts)
        
        assert isinstance(csv_output, str)
        assert "sku,forecast_date,prediction" in csv_output
        assert "WIDGET-001" in csv_output or "GADGET-002" in csv_output
        
        # Verify CSV structure
        lines = csv_output.strip().split('\n')
        assert len(lines) > 1  # Header + data rows
        
        # Check header
        header = lines[0]
        assert header == "sku,forecast_date,prediction"
    
    @pytest.mark.asyncio
    async def test_analyze_demand_patterns(self, demand_service, sample_demand_csv):
        """Test demand pattern analysis."""
        import pandas as pd
        from utils.helpers import read_csv_to_dict, validate_pydantic_data
        
        # Process CSV to DataFrame
        raw_data = read_csv_to_dict(sample_demand_csv, ['date', 'sku', 'quantity'])
        processed_data = []
        for row in raw_data:
            processed_data.append({
                'date': row['date'].strip(),
                'sku': row['sku'].strip().upper(),
                'quantity': int(row['quantity'])
            })
        
        validated_data = validate_pydantic_data(processed_data, DemandDataInput)
        df = pd.DataFrame([{
            'date': item.date,
            'sku': item.sku,
            'quantity': item.quantity
        } for item in validated_data])
        
        analytics_results = await demand_service.analyze_demand_patterns(df)
        
        assert isinstance(analytics_results, dict)
        assert len(analytics_results) > 0
        
        # Check analytics for first SKU
        first_sku = list(analytics_results.keys())[0]
        analytics = analytics_results[first_sku]
        
        assert analytics.sku == first_sku
        assert analytics.total_demand >= 0
        assert analytics.average_daily_demand >= 0
        assert analytics.demand_variance >= 0
        assert analytics.demand_std_dev >= 0
        assert analytics.coefficient_of_variation >= 0
        assert analytics.trend_direction in [trend.value for trend in TrendDirection]
        assert analytics.seasonality_pattern in [pattern.value for pattern in SeasonalityPattern]
        assert 0 <= analytics.data_quality_score <= 1
    
    @pytest.mark.asyncio
    async def test_get_model_summary(self, demand_service, sample_demand_csv):
        """Test model summary retrieval."""
        # First train a model
        await demand_service.process_demand_csv(sample_demand_csv, forecast_horizon_weeks=2)
        
        # Get model summary for first trained SKU
        trained_skus = list(demand_service.trained_models.keys())
        assert len(trained_skus) > 0
        
        sku = trained_skus[0]
        summary = await demand_service.get_model_summary(sku)
        
        assert summary is not None
        assert summary['sku'] == sku
        assert summary['model_version'] == demand_service.model_version
        assert 'training_date' in summary
        assert 'training_samples' in summary
        assert 'validation_samples' in summary
        assert 'performance_metrics' in summary
        assert 'feature_count' in summary
        assert 'top_features' in summary
        
        # Check performance metrics
        perf = summary['performance_metrics']
        assert 'train_mae' in perf
        assert 'val_mae' in perf
        assert 'val_mape' in perf
    
    @pytest.mark.asyncio
    async def test_get_model_summary_not_found(self, demand_service):
        """Test model summary for non-existent SKU."""
        summary = await demand_service.get_model_summary("NONEXISTENT-001")
        assert summary is None


# Convenience Function Tests

class TestConvenienceFunctions:
    """Test class for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_process_demand_csv_file(self, sample_demand_csv):
        """Test demand CSV file processing convenience function."""
        csv_output, summary = await process_demand_csv_file(sample_demand_csv, forecast_horizon_weeks=4)
        
        assert isinstance(csv_output, str)
        assert isinstance(summary, dict)
        assert summary["mode"] == "demand_forecasting"
        assert summary["total_forecasts"] > 0
        assert "generated_at" in summary
        assert "model_performance" in summary
        assert summary["forecast_horizon_weeks"] == 4
    
    @pytest.mark.asyncio
    async def test_analyze_demand_data(self, sample_demand_csv):
        """Test demand data analysis convenience function."""
        analytics_results, summary = await analyze_demand_data(sample_demand_csv)
        
        assert isinstance(analytics_results, dict)
        assert isinstance(summary, dict)
        assert len(analytics_results) > 0
        assert summary["total_skus_analyzed"] > 0
        assert "analysis_date" in summary
        assert "data_period_start" in summary
        assert "data_period_end" in summary
        
        # Check analytics structure
        first_sku = list(analytics_results.keys())[0]
        analytics = analytics_results[first_sku]
        
        assert 'total_demand' in analytics
        assert 'average_daily_demand' in analytics
        assert 'coefficient_of_variation' in analytics
        assert 'trend_direction' in analytics
        assert 'seasonality_pattern' in analytics
        assert 'data_quality_score' in analytics


# Data Model Tests

class TestDataModels:
    """Test class for data model validation."""
    
    def test_demand_data_input_validation(self):
        """Test demand data input model validation."""
        # Valid data
        valid_data = {
            "date": "2024-01-01",
            "sku": "WIDGET-001",
            "quantity": 150
        }
        
        item = DemandDataInput(**valid_data)
        assert item.date == "2024-01-01"
        assert item.sku == "WIDGET-001"
        assert item.quantity == 150
    
    def test_demand_data_input_validation_errors(self):
        """Test demand data input validation errors."""
        # Invalid date format
        with pytest.raises(ValueError):
            DemandDataInput(
                date="invalid-date",
                sku="WIDGET-001",
                quantity=100
            )
        
        # Negative quantity
        with pytest.raises(ValueError):
            DemandDataInput(
                date="2024-01-01",
                sku="WIDGET-001",
                quantity=-10
            )
        
        # Excessive quantity
        with pytest.raises(ValueError):
            DemandDataInput(
                date="2024-01-01",
                sku="WIDGET-001",
                quantity=2000000  # Exceeds 1M limit
            )
    
    def test_forecast_output_validation(self):
        """Test forecast output model validation."""
        # Valid data
        valid_data = {
            "sku": "WIDGET-001",
            "forecast_date": "2024-06-01",
            "prediction": 125.5,
            "confidence_interval_lower": 100.0,
            "confidence_interval_upper": 150.0,
            "confidence_level": 0.95,
            "forecast_accuracy": ForecastAccuracy.HIGH,
            "model_version": "v1.0"
        }
        
        forecast = ForecastOutput(**valid_data)
        assert forecast.sku == "WIDGET-001"
        assert forecast.prediction == 125.5
        assert forecast.confidence_level == 0.95
        assert forecast.forecast_accuracy == ForecastAccuracy.HIGH
    
    def test_forecast_output_validation_errors(self):
        """Test forecast output validation errors."""
        # Invalid confidence interval (lower > upper)
        with pytest.raises(ValueError):
            ForecastOutput(
                sku="WIDGET-001",
                forecast_date="2024-06-01",
                prediction=125.0,
                confidence_interval_lower=150.0,  # Higher than upper
                confidence_interval_upper=100.0,
                confidence_level=0.95,
                forecast_accuracy=ForecastAccuracy.HIGH,
                model_version="v1.0"
            )
        
        # Prediction outside confidence interval
        with pytest.raises(ValueError):
            ForecastOutput(
                sku="WIDGET-001",
                forecast_date="2024-06-01",
                prediction=200.0,  # Outside interval
                confidence_interval_lower=100.0,
                confidence_interval_upper=150.0,
                confidence_level=0.95,
                forecast_accuracy=ForecastAccuracy.HIGH,
                model_version="v1.0"
            )


# Error Handling Tests

class TestErrorHandling:
    """Test class for error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_csv_error(self, demand_service):
        """Test handling of empty CSV data."""
        with pytest.raises(CSVProcessingError):
            await demand_service.process_demand_csv("")
    
    @pytest.mark.asyncio
    async def test_malformed_csv_error(self, demand_service):
        """Test handling of malformed CSV data."""
        malformed_csv = "date,sku,quantity\n2024-01-01,WIDGET-001,not_a_number,extra_column"
        
        with pytest.raises((CSVProcessingError, DataValidationError)):
            await demand_service.process_demand_csv(malformed_csv)
    
    @pytest.mark.asyncio
    async def test_duplicate_date_sku_error(self, demand_service):
        """Test handling of duplicate date-SKU combinations."""
        duplicate_csv = """date,sku,quantity
2024-01-01,WIDGET-001,100
2024-01-01,WIDGET-001,110"""
        
        with pytest.raises(DataValidationError):
            await demand_service.process_demand_csv(duplicate_csv)
    
    @pytest.mark.asyncio
    async def test_future_date_error(self, demand_service):
        """Test handling of future dates in demand data."""
        future_date = (datetime.now() + timedelta(days=400)).strftime('%Y-%m-%d')
        future_csv = f"""date,sku,quantity
{future_date},WIDGET-001,100"""
        
        with pytest.raises(DataValidationError):
            await demand_service.process_demand_csv(future_csv)
    
    @pytest.mark.asyncio
    async def test_data_quality_error(self, demand_service):
        """Test handling of poor data quality."""
        # Create CSV with very poor quality (all zeros)
        poor_quality_csv = "date,sku,quantity\n"
        base_date = datetime.now() - timedelta(weeks=8)
        
        for day in range(56):
            current_date = base_date + timedelta(days=day)
            date_str = current_date.strftime('%Y-%m-%d')
            poor_quality_csv += f"{date_str},POOR-001,0\n"  # All zeros
        
        with pytest.raises(DataQualityException):
            await demand_service.process_demand_csv(poor_quality_csv)


# Performance Tests

class TestPerformance:
    """Test class for performance validation."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_processing(self, demand_service):
        """Test processing of larger datasets."""
        # Create larger CSV dataset (16 weeks, 5 SKUs)
        csv_lines = ["date,sku,quantity"]
        base_date = datetime.now() - timedelta(weeks=16)
        
        for day in range(112):  # 16 weeks * 7 days
            current_date = base_date + timedelta(days=day)
            date_str = current_date.strftime('%Y-%m-%d')
            
            for sku_num in range(5):
                sku = f"ITEM-{sku_num:03d}"
                demand = 100 + sku_num * 20 + (day % 7) * 5 + int((hash(date_str + sku) % 21) - 10)
                demand = max(0, demand)
                csv_lines.append(f"{date_str},{sku},{demand}")
        
        large_csv = '\n'.join(csv_lines)
        
        start_time = datetime.now()
        result = await demand_service.process_demand_csv(large_csv, forecast_horizon_weeks=4)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert result.total_forecasts > 0
        assert processing_time < 30.0  # Should process in under 30 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, demand_service, sample_demand_csv):
        """Test concurrent processing of multiple requests."""
        # Create multiple concurrent tasks
        tasks = []
        for i in range(3):  # Reduced from 5 due to ML processing overhead
            task = demand_service.process_demand_csv(sample_demand_csv, forecast_horizon_weeks=2)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        for result in results:
            assert result.total_forecasts > 0
            assert len(result.forecasts) > 0


# Machine Learning Tests

class TestMachineLearning:
    """Test class for ML-specific functionality."""
    
    @pytest.mark.asyncio
    async def test_model_caching(self, demand_service, sample_demand_csv):
        """Test that trained models are cached and reused."""
        # First training
        result1 = await demand_service.process_demand_csv(sample_demand_csv, forecast_horizon_weeks=2)
        initial_model_count = len(demand_service.trained_models)
        
        assert initial_model_count > 0
        
        # Second training with same data should reuse models
        result2 = await demand_service.process_demand_csv(sample_demand_csv, forecast_horizon_weeks=3)
        final_model_count = len(demand_service.trained_models)
        
        # Model count should be the same (models were reused)
        assert final_model_count == initial_model_count
        assert result2.total_forecasts > 0
    
    @pytest.mark.asyncio
    async def test_feature_importance(self, demand_service, sample_demand_csv):
        """Test that feature importance is calculated and stored."""
        await demand_service.process_demand_csv(sample_demand_csv, forecast_horizon_weeks=2)
        
        # Check that models have feature importance
        for sku in demand_service.trained_models.keys():
            model_summary = await demand_service.get_model_summary(sku)
            assert model_summary is not None
            
            perf = model_summary['performance_metrics']
            assert 'feature_importance' in perf
            assert len(perf['feature_importance']) > 0
            
            # Check that importance values are reasonable
            for feature, importance in perf['feature_importance'].items():
                assert isinstance(importance, (int, float))
                assert importance >= 0
    
    @pytest.mark.asyncio
    async def test_forecast_accuracy_classification(self, demand_service, sample_demand_csv):
        """Test that forecast accuracy is properly classified."""
        result = await demand_service.process_demand_csv(sample_demand_csv, forecast_horizon_weeks=2)
        
        # Check that all forecasts have valid accuracy classifications
        for forecast in result.forecasts:
            assert forecast.forecast_accuracy in [acc.value for acc in ForecastAccuracy]
            
            # Accuracy should be consistent with model performance
            # (This is a simplified check - in practice, accuracy depends on MAPE)
            assert forecast.forecast_accuracy in ["HIGH", "MEDIUM", "LOW"]


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])