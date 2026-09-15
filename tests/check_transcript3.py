import json

with open(r'C:\Users\shrey\.gemini\antigravity\brain\a002330a-ccfc-4407-80f6-b55a0ec3141d\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for i in range(5):
        line = f.readline()
        if not line: break
        obj = json.loads(line)
        print(f"Line {i}: type={obj.get('type')}, source={obj.get('source')}")
        c = obj.get('content', '')
        if isinstance(c, str):
            lines = c.splitlines()
            readings = [l for l in lines if l.startswith('reading_')]
            print(f"   total lines={len(lines)}, readings count={len(readings)}")
