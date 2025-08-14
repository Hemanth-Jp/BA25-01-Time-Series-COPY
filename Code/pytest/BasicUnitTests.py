"""
Basic Unit Testing with Pytest

This module demonstrates fundamental pytest concepts including:
- Simple test functions
- Basic assertions
- Test discovery patterns
- Grouping tests in classes


"""

import pytest
import math


# Simple test functions
def test_addition():
    """Test basic addition operation."""
    assert 1 + 1 == 2
    assert 10 + 5 == 15
    assert -3 + 7 == 4


def test_string_operations():
    """Test string manipulation functions."""
    text = "Hello, World!"
    assert text.upper() == "HELLO, WORLD!"
    assert text.lower() == "hello, world!"
    assert len(text) == 13
    assert "World" in text


def test_list_operations():
    """Test list operations and methods."""
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
    assert max(numbers) == 5
    assert min(numbers) == 1
    
    # Test list modification
    numbers.append(6)
    assert 6 in numbers
    assert numbers[-1] == 6


def test_mathematical_operations():
    """Test mathematical calculations."""
    assert math.sqrt(16) == 4.0
    assert math.pow(2, 3) == 8.0
    assert abs(-5) == 5
    assert round(3.14159, 2) == 3.14


# Test class for grouping related tests
class TestCalculator:
    """Test class demonstrating grouped test organization."""
    
    def test_divide_positive_numbers(self):
        """Test division with positive numbers."""
        assert 10 / 2 == 5.0
        assert 15 / 3 == 5.0
    
    def test_divide_by_zero_raises_exception(self):
        """Test that division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            10 / 0
    
    def test_multiply(self):
        """Test multiplication operations."""
        assert 3 * 4 == 12
        assert 0 * 100 == 0
        assert -2 * 5 == -10


class TestDataTypes:
    """Test class for different data type operations."""
    
    def test_dictionary_operations(self):
        """Test dictionary creation and manipulation."""
        data = {"name": "Alice", "age": 30}
        assert data["name"] == "Alice"
        assert "age" in data
        assert len(data) == 2
        
        # Add new key
        data["city"] = "New York"
        assert data["city"] == "New York"
    
    def test_set_operations(self):
        """Test set creation and operations."""
        numbers = {1, 2, 3, 4, 5}
        assert len(numbers) == 5
        assert 3 in numbers
        assert 6 not in numbers
        
        # Set operations
        more_numbers = {4, 5, 6, 7}
        intersection = numbers & more_numbers
        assert intersection == {4, 5}
    
    def test_tuple_immutability(self):
        """Test tuple properties and immutability."""
        coordinates = (10, 20)
        assert coordinates[0] == 10
        assert coordinates[1] == 20
        assert len(coordinates) == 2
        
        # Tuples are immutable
        with pytest.raises(TypeError):
            coordinates[0] = 5


# Parametrized test example
@pytest.mark.parametrize("input_value,expected", [
    (0, 0),
    (1, 1),
    (2, 4),
    (3, 9),
    (-2, 4),
])
def test_square_function(input_value, expected):
    """Test square function with multiple inputs."""
    assert input_value ** 2 == expected


# Testing edge cases
def test_empty_collections():
    """Test behavior with empty collections."""
    assert len([]) == 0
    assert len({}) == 0
    assert len(set()) == 0
    assert bool([]) is False
    assert bool({}) is False


def test_none_values():
    """Test handling of None values."""
    assert None is None
    assert None is not False
    assert None is not 0
    assert None is not ""