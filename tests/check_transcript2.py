import json

with open(r'C:\Users\shrey\.gemini\antigravity\brain\a002330a-ccfc-4407-80f6-b55a0ec3141d\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        keys = list(obj.keys())
        stype = obj.get('type')
        source = obj.get('source')
        print(f"Line {i}: type={stype}, source={source}, keys={keys}")
        if 'content' in obj:
            c = obj['content']
            if isinstance(c, str):
                print(f"   content len={len(c)}, first 50 chars: {repr(c[:50])}")
            elif isinstance(c, list):
                print(f"   content list len={len(c)}")
