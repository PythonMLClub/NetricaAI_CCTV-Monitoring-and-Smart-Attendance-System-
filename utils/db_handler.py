import pyodbc
from datetime import datetime
from dotenv import load_dotenv
import os
import logging

# Configure logging for error tracking (optional, remove in production if not needed)
logging.basicConfig(filename='database_errors.log', level=logging.ERROR)

# Load environment variables from .env file
load_dotenv()

# Table names as constants
Employees_table = "Employees"
AttendanceLogs_table = "AttendanceLogs"

# Get DB connection details from environment variables
def get_db_connection():
    # Load required environment variables
    db_driver = os.getenv('DB_DRIVER')
    db_server = os.getenv('DB_SERVER')
    db_name = os.getenv('DB_NAME')
    db_username = os.getenv('DB_USERNAME')
    db_password = os.getenv('DB_PASSWORD')

    # Check if all required variables are set
    missing_vars = []
    if not db_driver:
        missing_vars.append('DB_DRIVER')
    if not db_server:
        missing_vars.append('DB_SERVER')
    if not db_name:
        missing_vars.append('DB_NAME')
    if not db_username:
        missing_vars.append('DB_USERNAME')
    if not db_password:
        missing_vars.append('DB_PASSWORD')

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Create connection string
    conn_str = (
        f"DRIVER={db_driver};"  # Remove extra curly braces if driver name doesn't require them
        f"SERVER={db_server};"
        f"DATABASE={db_name};"
        f"UID={db_username};"
        f"PWD={db_password};"
        f"TrustServerCertificate=yes;"  # Optional: For self-signed certs in dev
    )
    try:
        return pyodbc.connect(conn_str)
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

def insert_employee(emp_id, full_name, department, role, shift_time, image_bytes, embedding_bytes=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if employee already exists
        cursor.execute(f"SELECT COUNT(*) FROM {Employees_table} WHERE EmployeeID = ?", (emp_id,))
        if cursor.fetchone()[0] > 0:
            return False, f"Employee ID {emp_id} already exists."

        # Insert new employee record (matching the table schema)
        query = f"""
            INSERT INTO {Employees_table} (EmployeeID, FullName, Department, Role, Shift, FaceImage, EmbeddingVector, CreatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(
            query,
            (emp_id, full_name, department, role, shift_time, pyodbc.Binary(image_bytes), 
             pyodbc.Binary(embedding_bytes) if embedding_bytes else None, datetime.now())
        )

        conn.commit()
        return True, "Employee registered successfully."

    except Exception as e:
        logging.error(f"Error inserting employee {emp_id}: {e}")
        return False, f"Database error: {e}"
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()


# # Fetch all employees who need embeddings
# def get_employees_without_embedding():
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         print("✅ Successfully connected to the database.")

#         cursor.execute(f"SELECT EmployeeID, FaceImage FROM {Employees_table};")
#         records = cursor.fetchall()
#         print(f"Fetched {len(records)} records without embeddings.")
#         return [(row[0], row[1]) for row in records]

#     except Exception as e:
#         logging.error(f"Error fetching employees without embeddings: {e}")
#         print(f"❌ Database error: {e}")
#         return []
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         if 'conn' in locals():
#             conn.close()

# # Update a specific employee's embedding
# def update_embedding(emp_id, embedding):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         cursor.execute(
#             f"UPDATE {Employees_table} SET EmbeddingVector = ? WHERE EmployeeID = ?",
#             (embedding.tobytes(), emp_id)
#         )
#         conn.commit()
#         return True, "Embedding updated successfully"

#     except Exception as e:
#         logging.error(f"Error updating embedding for {emp_id}: {e}")
#         return False, f"Error updating embedding: {e}"
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         if 'conn' in locals():
#             conn.close()

# # Get all embeddings with EmployeeIDs
# def get_all_embeddings():
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         cursor.execute(f"SELECT EmployeeID, EmbeddingVector FROM {Employees_table} WHERE EmbeddingVector IS NOT NULL")
#         records = cursor.fetchall()
#         return [(row[0], row[1]) for row in records]

#     except Exception as e:
#         logging.error(f"Error fetching embeddings: {e}")
#         print(f"❌ Database error: {e}")
#         return []
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         if 'conn' in locals():
#             conn.close()

# # Insert attendance logs
# def log_attendance(emp_id, timestamp, camera_id, location, confidence_score, event_type="Entry"):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         print(f"Attempting to log attendance for Employee ID: {emp_id}, Event: {event_type}")

#         cursor.execute(f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{AttendanceLogs_table}'")
#         if cursor.fetchone()[0] == 0:
#             return False, f"Table {AttendanceLogs_table} does not exist in the database."

#         query = f"""
#             INSERT INTO {AttendanceLogs_table} (EmployeeID, Timestamp, CameraID, Location, ConfidenceScore, Status)
#             VALUES (?, ?, ?, ?, ?, ?)
#         """
#         cursor.execute(query, (emp_id, timestamp, camera_id, location, confidence_score, event_type))
#         conn.commit()
#         print(f"Successfully logged attendance for Employee ID: {emp_id}")
#         return True, "Attendance logged successfully."

#     except Exception as e:
#         logging.error(f"Error logging attendance for {emp_id}: {e}")
#         print(f"❌ Database error while logging attendance: {e}")
#         return False, f"Database error: {e}"
#     finally:
#         if 'cursor' in locals(): cursor.close()
#         if 'conn' in locals(): conn.close()

# # Execute a generic query
# def execute_query(query, params=()):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute(query, params)
#         result = cursor.fetchall()
#         conn.commit()
#         return result
#     except Exception as e:
#         logging.error(f"Error executing query: {e}")
#         print(f"❌ Database error: {e}")
#         return []
#     finally:
#         if 'cursor' in locals(): cursor.close()
#         if 'conn' in locals(): conn.close()

# Get employee details
def get_employee_details(emp_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = f"SELECT FullName, Department, Role FROM {Employees_table} WHERE EmployeeID = ?"
        cursor.execute(query, (emp_id,))
        result = cursor.fetchone()  # Use fetchone() since we're expecting one row
        if result:
            return result[0], result[1], result[2]
        return None, None, None
    except Exception as e:
        logging.error(f"Error fetching details for {emp_id}: {e}")
        print(f"❌ Error fetching details for {emp_id}: {e}")
        return None, None, None
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# Example usage (for testing)
if __name__ == "__main__":
    try:
        # Test database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        print(f"Connected successfully: {cursor.fetchone()[0]}")
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Test failed: {e}")