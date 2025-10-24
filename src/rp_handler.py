import json
import requests

def run_tests(test_input):
    url = "http://localhost:8080"
    headers = {"Content-Type": "application/json"}
    
    input_data = json.loads(test_input)["input"]
    endpoint = f"{url}/generate/image"
    response = requests.post(endpoint, json=input_data, headers=headers, timeout=10000)
    
    if response.status_code != 200:
        raise Exception(f"Test failed: {response.text}")
    print(f"Test passed: {response.json()['message']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_tests(sys.argv[1])