import requests
import time

URL = "http://localhost:8000"

def main():
    print(f"=== EMPAS Serving Test Client ===")
    
    # 1. Health Check
    try:
        resp = requests.get(f"{URL}/health")
        print(f"Health Status: {resp.json()}")
    except Exception as e:
        print(f"Failed to connect to {URL}. Is the server running?")
        return

    # 2. Generation Request
    prompt = "The future of Artificial Intelligence is"
    payload = {"prompt": prompt, "max_length": 64}
    
    print(f"\nSending Prompt: '{prompt}'...")
    start = time.time()
    resp = requests.post(f"{URL}/generate", json=payload)
    end = time.time()
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"\n[Response (Latency: {data['latency_ms']:.2f}ms)]")
        print("-" * 40)
        print(data['generated_text'])
        print("-" * 40)
    else:
        print(f"Error: {resp.text}")

if __name__ == "__main__":
    main()