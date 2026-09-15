import os, string

drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
print("Available drives:", drives)

for d in drives:
    for root, dirs, files in os.walk(d):
        # search shallow or specific keywords
        if any(x in root for x in ['$Recycle.Bin', 'Windows', 'Program Files', 'AppData', '.git', 'node_modules']):
            continue
        for f in files:
            if 'energy' in f.lower() and f.endswith('.csv'):
                p = os.path.join(root, f)
                try:
                    with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                        lines = len(fp.readlines())
                    print(f"MATCH: {p} ({lines} lines)")
                except Exception as e:
                    print(f"MATCH (unreadable): {p}")
        if root.count(os.sep) > 4:
            dirs.clear() # limit depth
