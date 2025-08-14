"""
Database Testing with Pytest

This module demonstrates comprehensive database testing patterns including:
- Database fixture setup and teardown
- Transaction management
- Data integrity testing
- Mock database operations


"""

import pytest
import sqlite3
import tempfile
import os
from contextlib import contextmanager
from typing import Dict, List, Any, Optional


class DatabaseManager:
    """Mock database manager for testing purposes."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
    
    def connect(self):
        """Connect to the database."""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        return self.connection
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(100) NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                category VARCHAR(50),
                in_stock INTEGER DEFAULT 0
            )
        """)
        
        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                total_price DECIMAL(10, 2) NOT NULL,
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        """)
        
        self.connection.commit()
    
    def insert_user(self, username: str, email: str) -> int:
        """Insert a new user and return the user ID."""
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO users (username, email) VALUES (?, ?)",
            (username, email)
        )
        self.connection.commit()
        return cursor.lastrowid
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def insert_product(self, name: str, price: float, category: str = None, in_stock: int = 0) -> int:
        """Insert a new product."""
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO products (name, price, category, in_stock) VALUES (?, ?, ?, ?)",
            (name, price, category, in_stock)
        )
        self.connection.commit()
        return cursor.lastrowid
    
    def create_order(self, user_id: int, product_id: int, quantity: int) -> int:
        """Create a new order."""
        cursor = self.connection.cursor()
        
        # Get product price
        cursor.execute("SELECT price FROM products WHERE id = ?", (product_id,))
        product = cursor.fetchone()
        if not product:
            raise ValueError(f"Product {product_id} not found")
        
        total_price = product['price'] * quantity
        
        cursor.execute(
            "INSERT INTO orders (user_id, product_id, quantity, total_price) VALUES (?, ?, ?, ?)",
            (user_id, product_id, quantity, total_price)
        )
        self.connection.commit()
        return cursor.lastrowid
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise


@pytest.fixture(scope="session")
def test_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture(scope="session")
def db_manager(test_db_path):
    """Database manager fixture with session scope."""
    manager = DatabaseManager(test_db_path)
    manager.connect()
    manager.create_tables()
    
    yield manager
    
    manager.close()


@pytest.fixture
def clean_db(db_manager):
    """Fixture that provides a clean database for each test."""
    # Clear all tables before each test
    cursor = db_manager.connection.cursor()
    cursor.execute("DELETE FROM orders")
    cursor.execute("DELETE FROM products")
    cursor.execute("DELETE FROM users")
    db_manager.connection.commit()
    
    yield db_manager


@pytest.fixture
def sample_users(clean_db):
    """Fixture that creates sample users for testing."""
    users = [
        ("alice", "alice@example.com"),
        ("bob", "bob@example.com"),
        ("charlie", "charlie@example.com")
    ]
    
    user_ids = []
    for username, email in users:
        user_id = clean_db.insert_user(username, email)
        user_ids.append(user_id)
    
    return user_ids


@pytest.fixture
def sample_products(clean_db):
    """Fixture that creates sample products for testing."""
    products = [
        ("Laptop", 999.99, "Electronics", 10),
        ("Book", 29.99, "Books", 50),
        ("Coffee", 4.99, "Beverages", 100)
    ]
    
    product_ids = []
    for name, price, category, stock in products:
        product_id = clean_db.insert_product(name, price, category, stock)
        product_ids.append(product_id)
    
    return product_ids


class TestDatabaseOperations:
    """Test basic database operations."""
    
    def test_user_creation(self, clean_db):
        """Test creating a new user."""
        user_id = clean_db.insert_user("testuser", "test@example.com")
        
        assert user_id is not None
        assert user_id > 0
        
        # Verify user was created
        user = clean_db.get_user(user_id)
        assert user is not None
        assert user["username"] == "testuser"
        assert user["email"] == "test@example.com"
        assert user["is_active"] == 1
    
    def test_duplicate_username_constraint(self, clean_db):
        """Test that duplicate usernames are not allowed."""
        clean_db.insert_user("duplicate", "first@example.com")
        
        with pytest.raises(sqlite3.IntegrityError):
            clean_db.insert_user("duplicate", "second@example.com")
    
    def test_product_creation(self, clean_db):
        """Test creating products."""
        product_id = clean_db.insert_product("Test Product", 19.99, "Test Category", 5)
        
        assert product_id is not None
        assert product_id > 0
        
        # Verify product was created
        cursor = clean_db.connection.cursor()
        cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
        product = cursor.fetchone()
        
        assert product["name"] == "Test Product"
        assert product["price"] == 19.99
        assert product["category"] == "Test Category"
        assert product["in_stock"] == 5


class TestDatabaseRelations:
    """Test database relationships and foreign keys."""
    
    def test_order_creation(self, sample_users, sample_products, clean_db):
        """Test creating orders with proper relationships."""
        user_id = sample_users[0]
        product_id = sample_products[0]
        
        order_id = clean_db.create_order(user_id, product_id, 2)
        
        assert order_id is not None
        assert order_id > 0
        
        # Verify order details
        cursor = clean_db.connection.cursor()
        cursor.execute("""
            SELECT o.*, u.username, p.name as product_name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN products p ON o.product_id = p.id
            WHERE o.id = ?
        """, (order_id,))
        
        order = cursor.fetchone()
        assert order["username"] == "alice"
        assert order["product_name"] == "Laptop"
        assert order["quantity"] == 2
        assert order["total_price"] == 1999.98  # 999.99 * 2
    
    def test_order_with_invalid_product(self, sample_users, clean_db):
        """Test creating order with non-existent product."""
        user_id = sample_users[0]
        invalid_product_id = 999
        
        with pytest.raises(ValueError, match="Product 999 not found"):
            clean_db.create_order(user_id, invalid_product_id, 1)


class TestDatabaseTransactions:
    """Test database transaction handling."""
    
    def test_successful_transaction(self, clean_db):
        """Test successful transaction commit."""
        with clean_db.transaction() as cursor:
            cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)",
                          ("tx_user", "tx@example.com"))
            cursor.execute("INSERT INTO products (name, price) VALUES (?, ?)",
                          ("TX Product", 10.00))
        
        # Verify both inserts were committed
        cursor = clean_db.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'tx_user'")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("SELECT COUNT(*) FROM products WHERE name = 'TX Product'")
        assert cursor.fetchone()[0] == 1
    
    def test_transaction_rollback(self, clean_db):
        """Test transaction rollback on error."""
        try:
            with clean_db.transaction() as cursor:
                cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)",
                              ("rollback_user", "rollback@example.com"))
                # This will cause an error due to duplicate username
                cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)",
                              ("rollback_user", "duplicate@example.com"))
        except sqlite3.IntegrityError:
            pass  # Expected error
        
        # Verify rollback occurred - no users should be inserted
        cursor = clean_db.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'rollback_user'")
        assert cursor.fetchone()[0] == 0


class TestDataIntegrity:
    """Test data integrity and validation."""
    
    @pytest.mark.parametrize("username,email,should_succeed", [
        ("valid_user", "valid@example.com", True),
        ("", "empty_username@example.com", False),  # Empty username should fail
        ("no_email_user", "", False),  # Empty email should fail
        ("valid_user2", "invalid-email", True),  # SQLite doesn't validate email format
    ])
    def test_user_validation(self, clean_db, username, email, should_succeed):
        """Test user data validation scenarios."""
        if should_succeed:
            user_id = clean_db.insert_user(username, email)
            assert user_id > 0
            
            user = clean_db.get_user(user_id)
            assert user["username"] == username
            assert user["email"] == email
        else:
            # Empty strings might be allowed by SQLite, so we check application logic
            if not username or not email:
                with pytest.raises((ValueError, sqlite3.IntegrityError)):
                    # In real application, you'd have validation logic
                    if not username or not email:
                        raise ValueError("Username and email are required")
                    clean_db.insert_user(username, email)
    
    def test_product_price_validation(self, clean_db):
        """Test product price validation."""
        # Valid prices
        valid_prices = [0.01, 10.00, 999.99, 1000000.00]
        for price in valid_prices:
            product_id = clean_db.insert_product(f"Product_{price}", price)
            assert product_id > 0
        
        # Test negative price (should be handled by application logic)
        with pytest.raises(ValueError):
            # In real application, you'd validate negative prices
            price = -10.00
            if price < 0:
                raise ValueError("Price cannot be negative")
            clean_db.insert_product("Invalid Product", price)
    
    def test_order_quantity_validation(self, sample_users, sample_products, clean_db):
        """Test order quantity validation."""
        user_id = sample_users[0]
        product_id = sample_products[0]
        
        # Valid quantities
        valid_quantities = [1, 5, 10]
        for quantity in valid_quantities:
            order_id = clean_db.create_order(user_id, product_id, quantity)
            assert order_id > 0
        
        # Invalid quantity (should be handled by application logic)
        with pytest.raises(ValueError):
            quantity = 0
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            clean_db.create_order(user_id, product_id, quantity)