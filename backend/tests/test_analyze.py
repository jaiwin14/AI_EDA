import pytest
from fastapi.testclient import TestClient
import pandas as pd
import io
import numpy as np

@pytest.fixture
def test_data():
    """Create a test dataset"""
    data = {
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'purchased': [0, 1, 0, 1, 1],
        'date': pd.date_range('2023-01-01', periods=5).strftime('%Y-%m-%d')
    }
    df = pd.DataFrame(data)
    return df

def test_analyze_csv(test_data, client):
    """Test analysis of a CSV file"""
    # Convert DataFrame to CSV in memory
    csv_data = io.BytesIO()
    test_data.to_csv(csv_data, index=False)
    csv_data.seek(0)
    
    # Create a test file
    files = {'file': ('test.csv', csv_data, 'text/csv')}
    
    # Send request to the endpoint
    response = client.post("/api/v1/analyze/", files=files)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check basic structure
    assert 'filename' in data
    assert 'total_rows' in data
    assert 'total_columns' in data
    assert 'columns' in data
    
    # Check column analysis
    assert len(data['columns']) == 5  # age, income, gender, purchased, date
    
    # Check suggested target (should be 'purchased' for binary classification)
    assert data['suggested_target'] == 'purchased'
    assert data['suggested_task_type'] == 'binary_classification'

def test_analyze_excel(test_data, client):
    """Test analysis of an Excel file"""
    # Convert DataFrame to Excel in memory
    excel_data = io.BytesIO()
    with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
        test_data.to_excel(writer, index=False, sheet_name='Sheet1')
    excel_data.seek(0)
    
    # Create a test file
    files = {'file': ('test.xlsx', excel_data, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    
    # Send request to the endpoint
    response = client.post("/api/v1/analyze/", files=files)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check basic structure
    assert 'filename' in data
    assert 'total_rows' in data
    assert 'total_columns' in data
    assert 'columns' in data
    
    # Check column analysis
    assert len(data['columns']) == 5  # age, income, gender, purchased, date

def test_analyze_invalid_file(client):
    """Test analysis with an invalid file type"""
    # Create a test file with invalid content
    files = {'file': ('test.txt', io.BytesIO(b'invalid content'), 'text/plain')}
    
    # Send request to the endpoint
    response = client.post("/api/v1/analyze/", files=files)
    
    # Should return 400 for unsupported file type
    assert response.status_code == 400
    assert 'Unsupported file format' in response.json()['detail']
