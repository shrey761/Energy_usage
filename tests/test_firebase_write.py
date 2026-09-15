import urllib.request
import json

db_url = "https://voltix-1bac0-default-rtdb.asia-southeast1.firebasedatabase.app"

def test_firebase_write():
    url = f"{db_url}/sensor_logs/esp32_01/predictions/test.json"
    payload = {
        "status": "connected",
        "message": "AI ML Bridge verification"
    }
    try:
        data_bytes = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data_bytes, method='PUT')
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode())
            print("Successfully wrote to Firebase! Response:", res)
            return True
    except Exception as e:
        print("Firebase write error:", e)
        return False

if __name__ == "__main__":
    test_firebase_write()
