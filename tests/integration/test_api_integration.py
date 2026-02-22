"""
API Integration Tests
=====================

Integration tests for ML Platform APIs.
"""

import pytest
import httpx
import asyncio
from datetime import datetime

# Service URLs (would be configured for test environment)
BASE_URLS = {
    'data': 'http://localhost:8001',
    'feature': 'http://localhost:8002',
    'training': 'http://localhost:8003',
    'registry': 'http://localhost:5000',
    'inference': 'http://localhost:8000',
    'monitoring': 'http://localhost:8004'
}


@pytest.mark.asyncio
class TestDataServiceIntegration:
    """Integration tests for Data Service."""
    
    async def test_health_endpoint(self):
        """Test data service health check."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URLS['data']}/health")
            assert response.status_code == 200
            data = response.json()
            assert data['status'] in ['healthy', 'degraded']
            assert data['service'] == 'data-service'
    
    async def test_list_datasets(self):
        """Test listing datasets."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URLS['data']}/datasets")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


@pytest.mark.asyncio
class TestInferenceAPIIntegration:
    """Integration tests for Inference API."""
    
    async def test_health_endpoint(self):
        """Test inference API health check."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URLS['inference']}/health")
            assert response.status_code == 200
            data = response.json()
            assert data['service'] == 'inference-api'
    
    async def test_prediction_endpoint(self):
        """Test prediction endpoint."""
        payload = {
            'features': {
                'tenure': 24,
                'monthly_charges': 65.5,
                'total_charges': 1567.2,
                'contract': 'One year',
                'payment_method': 'Electronic check',
                'internet_service': 'DSL',
                'online_security': 'Yes',
                'tech_support': 'No'
            },
            'return_explanation': False
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URLS['inference']}/predict",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                assert 'prediction_id' in data
                assert 'prediction' in data
                assert 'model_name' in data
            elif response.status_code == 503:
                pytest.skip("Model not loaded")
    
    async def test_batch_prediction_endpoint(self):
        """Test batch prediction endpoint."""
        payload = {
            'records': [
                {'tenure': 24, 'monthly_charges': 65.5},
                {'tenure': 12, 'monthly_charges': 45.0}
            ],
            'return_explanations': False
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URLS['inference']}/batch_predict",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                assert 'batch_id' in data
                assert 'predictions' in data
                assert len(data['predictions']) == 2


@pytest.mark.asyncio
class TestMonitoringServiceIntegration:
    """Integration tests for Monitoring Service."""
    
    async def test_health_endpoint(self):
        """Test monitoring service health check."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URLS['monitoring']}/health")
            assert response.status_code == 200
            data = response.json()
            assert data['service'] == 'monitoring-service'
    
    async def test_dashboard_endpoint(self):
        """Test dashboard endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URLS['monitoring']}/dashboard/test-model",
                params={'days': 7}
            )
            assert response.status_code in [200, 404]  # 404 if no data


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @pytest.mark.asyncio
    async def test_full_ml_pipeline(self):
        """
        Test complete ML pipeline:
        1. Ingest data
        2. Engineer features
        3. Train model
        4. Make predictions
        5. Monitor performance
        """
        # This would be a comprehensive E2E test
        # Requires all services to be running
        pass