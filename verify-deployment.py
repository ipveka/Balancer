#!/usr/bin/env python3
"""
Deployment Verification Script for Balancer Platform

This script verifies that the Balancer platform is properly deployed
and all endpoints are accessible.
"""

import requests
import sys
import time
import json
from typing import Dict, Any


def test_endpoint(url: str, expected_status: int = 200, timeout: int = 10) -> Dict[str, Any]:
    """Test a single endpoint and return results."""
    try:
        response = requests.get(url, timeout=timeout)
        return {
            "url": url,
            "status_code": response.status_code,
            "success": response.status_code == expected_status,
            "response_time": response.elapsed.total_seconds(),
            "error": None
        }
    except Exception as e:
        return {
            "url": url,
            "status_code": None,
            "success": False,
            "response_time": None,
            "error": str(e)
        }


def verify_deployment(base_url: str = "http://localhost:8000") -> bool:
    """Verify the deployment by testing all critical endpoints."""
    
    print("🔍 Balancer Platform Deployment Verification")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    print()
    
    # Define endpoints to test
    endpoints = [
        {"path": "/", "name": "Root endpoint"},
        {"path": "/health", "name": "Health check"},
        {"path": "/api/v1/status", "name": "API status"},
        {"path": "/api/v1/modules", "name": "Modules list"},
        {"path": "/docs", "name": "API documentation"},
        {"path": "/openapi.json", "name": "OpenAPI schema"},
    ]
    
    results = []
    all_passed = True
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint['path']}"
        print(f"Testing {endpoint['name']}... ", end="")
        
        result = test_endpoint(url)
        results.append({**result, "name": endpoint["name"]})
        
        if result["success"]:
            print(f"✅ ({result['response_time']:.3f}s)")
        else:
            error_msg = result.get('error', f"Status: {result.get('status_code', 'N/A')}")
            print(f"❌ {error_msg}")
            all_passed = False
    
    print()
    print("📊 Test Results Summary")
    print("-" * 30)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        time_str = f" ({result['response_time']:.3f}s)" if result["response_time"] else ""
        print(f"{result['name']:<20} {status}{time_str}")
    
    print()
    
    if all_passed:
        print("🎉 All tests passed! Deployment is successful.")
        
        # Additional verification
        print("\n🔧 Additional Checks")
        print("-" * 20)
        
        # Test health endpoint response
        health_result = test_endpoint(f"{base_url}/health")
        if health_result["success"]:
            try:
                health_response = requests.get(f"{base_url}/health")
                health_data = health_response.json()
                print(f"✅ Health status: {health_data.get('status', 'unknown')}")
                print(f"✅ Version: {health_data.get('version', 'unknown')}")
                print(f"✅ Environment: {health_data.get('environment', 'unknown')}")
            except:
                print("⚠️  Could not parse health response")
        
        # Test modules endpoint
        modules_result = test_endpoint(f"{base_url}/api/v1/modules")
        if modules_result["success"]:
            try:
                modules_response = requests.get(f"{base_url}/api/v1/modules")
                modules_data = modules_response.json()
                module_count = modules_data.get('total_modules', 0)
                print(f"✅ Available modules: {module_count}")
            except:
                print("⚠️  Could not parse modules response")
        
        return True
    else:
        print("❌ Some tests failed. Please check the deployment.")
        return False


def main():
    """Main function to run deployment verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Balancer platform deployment")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="Base URL of the deployed application (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Wait time in seconds before starting verification (default: 0)"
    )
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds before verification...")
        time.sleep(args.wait)
    
    success = verify_deployment(args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()