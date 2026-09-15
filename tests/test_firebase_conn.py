import urllib.request
import json

db_url = "https://voltix-1bac0-default-rtdb.asia-southeast1.firebasedatabase.app"

def test_firebase_read():
    url = f"{db_url}/sensor_logs/esp32_01/readings.json?orderBy=%22$key%22&limitToLast=5"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            print("Successfully read from Firebase! Entries:", len(data) if data else 0)
            if data:
                for k, v in list(data.items())[-2:]:
                    print(f"  Key: {k}")
                    print(f"  Data: {v}")
            return data
    except Exception as e:
        print("Firebase read error:", e)
        return None

if __name__ == "__main__":
    test_firebase_read()
