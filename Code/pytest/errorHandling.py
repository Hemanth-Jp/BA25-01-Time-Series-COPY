"""
Comprehensive Error Handling with Pytest

This module demonstrates robust error handling patterns including:
- Exception testing with pytest.raises
- Custom exception handling
- Timeout and resource management
- Test failure debugging techniques

"""

import pytest
import time
import os
import tempfile
from contextlib import contextmanager
from typing import List, Dict, Any, Optional


class CustomError(Exception):
    """Custom exception for demonstration purposes."""
    pass


class ValidationError(Exception):
    """Exception raised for validation errors."""
    
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error in field '{field}': {message}")


class ResourceManager:
    """Mock resource manager for testing resource handling."""
    
    def __init__(self, resource_name: str):
        self.resource_name = resource_name
        self.is_connected = False
        self.operations_count = 0
    
    def connect(self):
        """Connect to resource."""
        if self.resource_name == "fail_connect":
            raise ConnectionError(f"Failed to connect to {self.resource_name}")
        self.is_connected = True
    
    def disconnect(self):
        """Disconnect from resource."""
        if not self.is_connected:
            raise RuntimeError("Not connected")
        self.is_connected = False
    
    def perform_operation(self, operation_type: str):
        """Perform an operation on the resource."""
        if not self.is_connected:
            raise RuntimeError("Resource not connected")
        
        if operation_type == "timeout":
            time.sleep(2)  # Simulate long operation
        elif operation_type == "error":
            raise CustomError("Operation failed")
        
        self.operations_count += 1
        return f"Operation {operation_type} completed"


def divide_numbers(a: float, b: float) -> float:
    """Division function that handles various error cases."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    
    return a / b


def validate_user_data(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate user data and raise specific validation errors."""
    if not isinstance(user_data, dict):
        raise TypeError("User data must be a dictionary")
    
    required_fields = ["name", "email", "age"]
    for field in required_fields:
        if field not in user_data:
            raise ValidationError(field, "Field is required")
        
        if not user_data[field]:
            raise ValidationError(field, "Field cannot be empty")
    
    # Age validation
    age = user_data["age"]
    if not isinstance(age, int) or age < 0 or age > 150:
        raise ValidationError("age", "Age must be a positive integer between 0 and 150")
    
    # Email validation (basic)
    email = user_data["email"]
    if "@" not in email or "." not in email:
        raise ValidationError("email", "Invalid email format")
    
    return user_data


def process_file(file_path: str) -> List[str]:
    """Process a file and handle various file-related errors."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Permission denied reading file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]
    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error: {e}")


class TestBasicExceptionHandling:
    """Test basic exception handling patterns."""
    
    def test_division_by_zero(self):
        """Test that division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            divide_numbers(10, 0)
    
    def test_division_by_zero_with_message(self):
        """Test exception with specific error message."""
        with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
            divide_numbers(5, 0)
    
    def test_type_error_in_division(self):
        """Test type error with invalid arguments."""
        with pytest.raises(TypeError, match="Arguments must be numbers"):
            divide_numbers("10", 5)
        
        with pytest.raises(TypeError):
            divide_numbers(10, "5")
    
    def test_successful_division(self):
        """Test successful division operations."""
        assert divide_numbers(10, 2) == 5.0
        assert divide_numbers(-10, 2) == -5.0
        assert divide_numbers(7, 3) == pytest.approx(2.333, rel=1e-2)


class TestCustomExceptionHandling:
    """Test custom exception handling."""
    
    def test_validation_error_missing_field(self):
        """Test validation error for missing required field."""
        incomplete_data = {"name": "John", "email": "john@example.com"}  # Missing age
        
        with pytest.raises(ValidationError) as exc_info:
            validate_user_data(incomplete_data)
        
        assert exc_info.value.field == "age"
        assert "required" in exc_info.value.message
    
    def test_validation_error_invalid_age(self):
        """Test validation error for invalid age."""
        invalid_data = {"name": "John", "email": "john@example.com", "age": -5}
        
        with pytest.raises(ValidationError) as exc_info:
            validate_user_data(invalid_data)
        
        assert exc_info.value.field == "age"
        assert "positive integer" in exc_info.value.message
    
    def test_validation_error_invalid_email(self):
        """Test validation error for invalid email format."""
        invalid_data = {"name": "John", "email": "invalid-email", "age": 25}
        
        with pytest.raises(ValidationError) as exc_info:
            validate_user_data(invalid_data)
        
        assert exc_info.value.field == "email"
        assert "Invalid email format" in exc_info.value.message
    
    def test_successful_validation(self):
        """Test successful data validation."""
        valid_data = {"name": "John Doe", "email": "john@example.com", "age": 30}
        result = validate_user_data(valid_data)
        assert result == valid_data


class TestResourceManagement:
    """Test resource management and cleanup scenarios."""
    
    def test_successful_resource_connection(self):
        """Test successful resource connection and disconnection."""
        manager = ResourceManager("test_resource")
        manager.connect()
        
        assert manager.is_connected
        
        result = manager.perform_operation("normal")
        assert "completed" in result
        assert manager.operations_count == 1
        
        manager.disconnect()
        assert not manager.is_connected
    
    def test_connection_failure(self):
        """Test handling of connection failures."""
        manager = ResourceManager("fail_connect")
        
        with pytest.raises(ConnectionError, match="Failed to connect"):
            manager.connect()
        
        assert not manager.is_connected
    
    def test_operation_without_connection(self):
        """Test operation failure when resource is not connected."""
        manager = ResourceManager("test_resource")
        
        with pytest.raises(RuntimeError, match="Resource not connected"):
            manager.perform_operation("normal")
    
    def test_disconnect_without_connection(self):
        """Test disconnect failure when not connected."""
        manager = ResourceManager("test_resource")
        
        with pytest.raises(RuntimeError, match="Not connected"):
            manager.disconnect()
    
    def test_operation_failure_handling(self):
        """Test handling of operation failures."""
        manager = ResourceManager("test_resource")
        manager.connect()
        
        with pytest.raises(CustomError, match="Operation failed"):
            manager.perform_operation("error")
        
        # Resource should still be connected after operation failure
        assert manager.is_connected
        
        # Should be able to perform other operations
        result = manager.perform_operation("normal")
        assert "completed" in result


class TestFileOperations:
    """Test file operation error handling."""
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            process_file("/path/to/nonexistent/file.txt")
    
    def test_successful_file_processing(self, tmp_path):
        """Test successful file processing."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_content = ["line 1", "line 2", "line 3"]
        test_file.write_text("\n".join(test_content))
        
        result = process_file(str(test_file))
        assert result == test_content
    
    def test_empty_file_processing(self, tmp_path):
        """Test processing of empty files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        result = process_file(str(empty_file))
        assert result == []


class TestTimeoutAndPerformance:
    """Test timeout scenarios and performance constraints."""
    
    @pytest.mark.timeout(1)
    def test_fast_operation(self):
        """Test that operation completes within timeout."""
        manager = ResourceManager("fast_resource")
        manager.connect()
        result = manager.perform_operation("normal")
        assert "completed" in result
    
    def test_timeout_handling_with_pytest_raises(self):
        """Test timeout handling using pytest.raises."""
        # This would require pytest-timeout plugin in real scenario
        manager = ResourceManager("slow_resource")
        manager.connect()
        
        # Simulate timeout by checking operation duration
        start_time = time.time()
        try:
            manager.perform_operation("timeout")  # This takes 2 seconds
        except Exception:
            pass
        duration = time.time() - start_time
        
        # In real scenario, this would be handled by timeout decorator
        assert duration >= 2.0


class TestParametrizedErrorHandling:
    """Test error handling with parametrized tests."""
    
    @pytest.mark.parametrize("input_data,expected_exception", [
        (None, TypeError),
        ("not a dict", TypeError),
        ({}, ValidationError),
        ({"name": ""}, ValidationError),
        ({"name": "John"}, ValidationError),  # Missing email and age
    ])
    def test_validation_errors(self, input_data, expected_exception):
        """Test various validation error scenarios."""
        with pytest.raises(expected_exception):
            validate_user_data(input_data)
    
    @pytest.mark.parametrize("a,b,expected_exception", [
        ("string", 5, TypeError),
        (10, "string", TypeError),
        (10, 0, ZeroDivisionError),
        (None, 5, TypeError),
    ])
    def test_division_errors(self, a, b, expected_exception):
        """Test various division error scenarios."""
        with pytest.raises(expected_exception):
            divide_numbers(a, b)