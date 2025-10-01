#!/usr/bin/env python3
"""
Test script to verify correlation analysis data structure
"""
import requests
import json

def test_correlation_endpoint():
    """Test the correlation analysis endpoint"""
    try:
        # Replace with your actual dataset ID
        dataset_id = "65fc9451-aa84-416d-a465-61b28e0f3228"
        url = f"http://localhost:8000/api/v1/ai-insights/{dataset_id}/correlations"
        
        print(f"Testing URL: {url}")
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Request successful!")
            print(f"Response structure:")
            print(json.dumps(data, indent=2))
            
            # Check data structure
            if 'data' in data:
                data_section = data['data']
                print(f"\n📊 Data analysis:")
                print(f"- Correlations found: {len(data_section.get('correlations', []))}")
                print(f"- Insights found: {len(data_section.get('insights', []))}")
                print(f"- Strong correlations: {len(data_section.get('strong_correlations', []))}")
                print(f"- Matrix keys: {len(data_section.get('correlation_matrix', {}))}")
                
                if data_section.get('correlations'):
                    print(f"\n🔗 Sample correlations:")
                    for i, corr in enumerate(data_section['correlations'][:3]):
                        print(f"  {i+1}. {corr['var1']} ↔ {corr['var2']}: {corr['correlation']}")
            else:
                print("❌ No 'data' field found in response")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the backend is running on port 8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_correlation_endpoint()
