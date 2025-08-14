"""
API Testing with Pytest

This module demonstrates comprehensive API testing patterns including:
- HTTP client fixtures
- Request/response validation
- Authentication testing
- Error handling and status codes


"""

import pytest
import json
import requests
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional


class APIClient:
    """Mock API client for testing purposes."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Mock GET request."""
        url = f"{self.base_url}{endpoint}"
        
        # Simulate different responses based on endpoint
        if endpoint == "/users":
            return {
                "status_code": 200,
                "data": [
                    {"id": 1, "name": "Alice", "email": "alice@example.com"},
                    {"id": 2, "name": "Bob", "email": "bob@example.com"}
                ]
            }
        elif endpoint == "/users/1":
            return {
                "status_code": 200,
                "data": {"id": 1, "name": "Alice", "email": "alice@example.com"}
            }
        elif endpoint == "/users/999":
            return {"status_code": 404, "error": "User not found"}
        else:
            return {"status_code": 200, "data": {}}
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock POST request."""
        if endpoint == "/users":
            if not data.get("name") or not data.get("email"):
                return {"status_code": 400, "error": "Missing required fields"}
            
            new_user = {
                "id": 3,
                "name": data["name"],
                "email": data["email"]
            }
            return {"status_code": 201, "data": new_user}
        
        return {"status_code": 200, "data": data}
    
    def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock PUT request."""
        if "/users/" in endpoint:
            return {"status_code": 200, "data": data}
        return {"status_code": 404, "error": "Resource not found"}
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Mock DELETE request."""
        if endpoint == "/users/1":
            return {"status_code": 204, "data": None}
        return {"status_code": 404, "error": "Resource not found"}


@pytest.fixture(scope="session")
def api_base_url():
    """Base URL for API testing."""
    return "https://api.example.com"


@pytest.fixture(scope="session")
def api_key():
    """API key for authentication."""
    return "test-api-key-12345"


@pytest.fixture
def api_client(api_base_url, api_key):
    """API client fixture with authentication."""
    return APIClient(api_base_url, api_key)


@pytest.fixture
def unauthenticated_client(api_base_url):
    """API client without authentication."""
    return APIClient(api_base_url)


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "role": "user"
    }


class TestAPIAuthentication:
    """Test API authentication mechanisms."""
    
    def test_authenticated_request(self, api_client):
        """Test that authenticated requests work properly."""
        response = api_client.get("/users")
        assert response["status_code"] == 200
        assert "data" in response
    
    def test_unauthenticated_request(self, unauthenticated_client):
        """Test unauthenticated requests."""
        # In real scenario, this might return 401
        response = unauthenticated_client.get("/users")
        assert response["status_code"] == 200  # Mock always returns 200
    
    def test_invalid_api_key(self, api_base_url):
        """Test requests with invalid API key."""
        client = APIClient(api_base_url, "invalid-key")
        response = client.get("/users")
        # In real scenario, this would return 401
        assert response["status_code"] == 200


class TestUserAPI:
    """Test user-related API endpoints."""
    
    def test_get_all_users(self, api_client):
        """Test retrieving all users."""
        response = api_client.get("/users")
        
        assert response["status_code"] == 200
        assert isinstance(response["data"], list)
        assert len(response["data"]) == 2
        
        # Validate user structure
        user = response["data"][0]
        assert "id" in user
        assert "name" in user
        assert "email" in user
    
    def test_get_user_by_id(self, api_client):
        """Test retrieving a specific user."""
        response = api_client.get("/users/1")
        
        assert response["status_code"] == 200
        assert response["data"]["id"] == 1
        assert response["data"]["name"] == "Alice"
    
    def test_get_nonexistent_user(self, api_client):
        """Test retrieving a non-existent user."""
        response = api_client.get("/users/999")
        
        assert response["status_code"] == 404
        assert "error" in response
    
    def test_create_user(self, api_client, sample_user_data):
        """Test creating a new user."""
        response = api_client.post("/users", sample_user_data)
        
        assert response["status_code"] == 201
        assert response["data"]["name"] == sample_user_data["name"]
        assert response["data"]["email"] == sample_user_data["email"]
        assert "id" in response["data"]
    
    def test_create_user_missing_data(self, api_client):
        """Test creating user with missing required fields."""
        incomplete_data = {"name": "John"}  # Missing email
        response = api_client.post("/users", incomplete_data)
        
        assert response["status_code"] == 400
        assert "error" in response
    
    def test_update_user(self, api_client):
        """Test updating an existing user."""
        update_data = {"name": "Alice Updated", "email": "alice.updated@example.com"}
        response = api_client.put("/users/1", update_data)
        
        assert response["status_code"] == 200
        assert response["data"]["name"] == update_data["name"]
    
    def test_delete_user(self, api_client):
        """Test deleting a user."""
        response = api_client.delete("/users/1")
        
        assert response["status_code"] == 204
        assert response["data"] is None


class TestAPIErrorHandling:
    """Test API error handling scenarios."""
    
    @pytest.mark.parametrize("endpoint,expected_status", [
        ("/users/1", 200),
        ("/users/999", 404),
        ("/nonexistent", 200),  # Mock returns 200 for unknown endpoints
    ])
    def test_various_endpoints_status_codes(self, api_client, endpoint, expected_status):
        """Test different endpoints return appropriate status codes."""
        response = api_client.get(endpoint)
        assert response["status_code"] == expected_status
    
    @pytest.mark.parametrize("user_data,expected_status", [
        ({"name": "Valid User", "email": "valid@example.com"}, 201),
        ({"name": "No Email User"}, 400),
        ({"email": "nousername@example.com"}, 400),
        ({}, 400),
    ])
    def test_user_creation_validation(self, api_client, user_data, expected_status):
        """Test user creation with various data validation scenarios."""
        response = api_client.post("/users", user_data)
        assert response["status_code"] == expected_status
    
    def test_request_timeout_handling(self, api_client):
        """Test handling of request timeouts."""
        # In a real scenario, you might mock requests.exceptions.Timeout
        with patch.object(api_client.session, 'get', side_effect=requests.exceptions.Timeout):
            with pytest.raises(requests.exceptions.Timeout):
                api_client.session.get("/users")
    
    def test_connection_error_handling(self, api_client):
        """Test handling of connection errors."""
        with patch.object(api_client.session, 'get', side_effect=requests.exceptions.ConnectionError):
            with pytest.raises(requests.exceptions.ConnectionError):
                api_client.session.get("/users")