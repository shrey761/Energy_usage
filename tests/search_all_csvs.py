import os

search_paths = [
    r"C:\Users\shrey",
    r"C:\Users\shrey\Downloads",
    r"C:\Users\shrey\Desktop",
    r"C:\Users\shrey\Documents",
    r"C:\Users\shrey\OneDrive",
    r"C:\Users\shrey\gitrepo",
    r"C:\Users\shrey\.gemini",
]

found = []
for p in search_paths:
    if not os.path.exists(p): continue
    for root, dirs, files in os.walk(p):
        if any(x in root for x in ['.git', 'node_modules', 'venv', '.venv', 'AppData']):
            continue
        for f in files:
            if f.endswith('.csv'):
                full_path = os.path.join(root, f)
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as fp:
                        lines = len(fp.readlines())
                    found.append((full_path, lines))
                except Exception as e:
                    pass

print(f"Total CSVs found: {len(found)}")
for path, lines in found:
    print(f"  {path} : {lines} lines")
