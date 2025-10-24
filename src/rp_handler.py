import json
import requests
import argparse
import os

def run_tests(test_input):
    url = "http://localhost:8080"
    headers = {"Content-Type": "application/json"}
    
    input_data = json.loads(test_input)["input"]
    endpoint = f"{url}/generate/{input_data['type']}"
    response = requests.post(endpoint, json=input_data, headers=headers, timeout=10000)
    
    if response.status_code != 200:
        raise Exception(f"Test failed: {response.text}")
    print(f"Test passed for {input_data['type']}: {response.json()['message']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_input", type=str, required=True)
    args = parser.parse_args()
    run_tests(args.test_input)
