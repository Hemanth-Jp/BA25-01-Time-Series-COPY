"""
Advanced Pytest Features

This module demonstrates advanced pytest capabilities including:
- Complex fixtures with different scopes
- Parametrization with multiple parameters
- Fixture dependencies and composition
- Custom markers and test categorization


"""

import pytest
import tempfile
import json
import os
from datetime import datetime
from typing import Dict, List, Any


# Session-scoped fixture for expensive setup
@pytest.fixture(scope="session")
def global_config():
    """Global configuration fixture with session scope."""
    config = {
        "api_url": "https://api.example.com",
        "timeout": 30,
        "retry_count": 3,
        "environment": "test"
    }
    print(f"\nSetting up global config: {config}")
    yield config
    print("\nTearing down global config")


# Module-scoped fixture
@pytest.fixture(scope="module")
def test_database():
    """Module-scoped database fixture."""
    db_name = f"test_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nCreating test database: {db_name}")
    
    # Simulate database creation
    database = {
        "name": db_name,
        "tables": ["users", "products", "orders"],
        "connection": f"sqlite:///{db_name}.db"
    }
    
    yield database
    
    print(f"\nCleaning up database: {db_name}")


# Function-scoped fixture with dependency
@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test_data.json"
    test_data = {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ]
    }
    
    with open(file_path, 'w') as f:
        json.dump(test_data, f)
    
    yield file_path
    
    # Cleanup happens automatically with tmp_path


# Parametrized fixture
@pytest.fixture(params=["sqlite", "postgresql", "mysql"])
def database_type(request):
    """Parametrized fixture for different database types."""
    return request.param


# Fixture with indirect parametrization
@pytest.fixture
def user_data(request):
    """Fixture that creates user data based on parameters."""
    if hasattr(request, 'param'):
        user_type = request.param
    else:
        user_type = "regular"
    
    users = {
        "admin": {"name": "Admin", "role": "administrator", "permissions": ["read", "write", "delete"]},
        "regular": {"name": "User", "role": "user", "permissions": ["read"]},
        "guest": {"name": "Guest", "role": "guest", "permissions": []}
    }
    
    return users.get(user_type, users["regular"])


# Complex fixture composition
@pytest.fixture
def api_client(global_config):
    """API client fixture that depends on global config."""
    class MockAPIClient:
        def __init__(self, config):
            self.base_url = config["api_url"]
            self.timeout = config["timeout"]
            self.session_id = f"session_{datetime.now().timestamp()}"
        
        def get(self, endpoint):
            return {"status": "success", "data": f"GET {endpoint}"}
        
        def post(self, endpoint, data):
            return {"status": "success", "data": f"POST {endpoint}", "payload": data}
    
    client = MockAPIClient(global_config)
    yield client


class TestFixtureScopes:
    """Test class demonstrating fixture scopes."""
    
    def test_session_fixture_usage(self, global_config):
        """Test using session-scoped fixture."""
        assert global_config["environment"] == "test"
        assert "api_url" in global_config
    
    def test_module_fixture_usage(self, test_database):
        """Test using module-scoped fixture."""
        assert "test_db_" in test_database["name"]
        assert len(test_database["tables"]) == 3
    
    def test_function_fixture_usage(self, temp_file):
        """Test using function-scoped fixture."""
        assert temp_file.exists()
        with open(temp_file, 'r') as f:
            data = json.load(f)
        assert len(data["users"]) == 2


class TestParametrization:
    """Test class demonstrating various parametrization techniques."""
    
    @pytest.mark.parametrize("database_type", ["sqlite", "postgresql"], indirect=True)
    def test_database_connection(self, database_type):
        """Test database connections with different types."""
        assert database_type in ["sqlite", "postgresql", "mysql"]
    
    @pytest.mark.parametrize("user_data", ["admin", "regular", "guest"], indirect=True)
    def test_user_permissions(self, user_data):
        """Test user permissions for different user types."""
        assert "permissions" in user_data
        assert isinstance(user_data["permissions"], list)
    
    @pytest.mark.parametrize("input_data,expected_status", [
        ({"name": "test"}, "success"),
        ({}, "error"),
        ({"name": ""}, "error"),
        ({"name": "valid", "email": "test@example.com"}, "success")
    ])
    def test_data_validation(self, input_data, expected_status):
        """Test data validation with multiple scenarios."""
        # Simulate validation logic
        if input_data.get("name") and input_data["name"].strip():
            status = "success"
        else:
            status = "error"
        
        assert status == expected_status


# Custom markers for test categorization
@pytest.mark.slow
def test_slow_operation(api_client):
    """Test marked as slow for selective execution."""
    import time
    time.sleep(0.1)  # Simulate slow operation
    result = api_client.get("/slow-endpoint")
    assert result["status"] == "success"


@pytest.mark.integration
def test_api_integration(api_client, global_config):
    """Integration test using multiple fixtures."""
    # Test API endpoint
    response = api_client.get("/users")
    assert response["status"] == "success"
    
    # Test configuration
    assert api_client.base_url == global_config["api_url"]


@pytest.mark.parametrize("method,endpoint,data", [
    ("get", "/users", None),
    ("post", "/users", {"name": "New User"}),
    ("get", "/products", None),
])
def test_api_methods(api_client, method, endpoint, data):
    """Test different API methods and endpoints."""
    if method == "get":
        response = api_client.get(endpoint)
    elif method == "post":
        response = api_client.post(endpoint, data)
    
    assert response["status"] == "success"
    assert endpoint in response["data"]