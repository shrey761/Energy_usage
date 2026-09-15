import json

with open(r'C:\Users\shrey\.gemini\antigravity\brain\a002330a-ccfc-4407-80f6-b55a0ec3141d\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        content_str = json.dumps(obj)
        if 'reading_1003718' in content_str:
            print(f"Found in line {i}!")
            for k, v in obj.items():
                if isinstance(v, str) and 'reading_1003718' in v:
                    print(f"  key {k}: string of length {len(v)}, lines={len(v.splitlines())}")
                elif isinstance(v, (list, dict)):
                    s = json.dumps(v)
                    if 'reading_1003718' in s:
                        print(f"  key {k}: type {type(v)}, sub-string len={len(s)}")
