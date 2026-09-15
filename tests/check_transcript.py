import json

with open(r'C:\Users\shrey\.gemini\antigravity\brain\a002330a-ccfc-4407-80f6-b55a0ec3141d\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        content = obj.get('content', '')
        if 'reading_1003718' in content:
            print(f"Step {i}: type={obj.get('type')}, lines={len(content.splitlines())}")
            # Count reading_ lines
            readings = [l for l in content.splitlines() if l.startswith('reading_')]
            print(f"Reading count in step {i}: {len(readings)}")
