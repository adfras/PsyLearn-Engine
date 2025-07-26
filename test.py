import os

def get_file_sizes(root_dir, limit=50):
    all_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            path = os.path.join(root, f)
            try:
                size = os.path.getsize(path)
                all_files.append((path, size))
            except Exception:
                continue

    all_files.sort(key=lambda x: -x[1])
    for path, size in all_files[:limit]:
        print(f"{size/1e6:6.2f} MB  {path}")

print("--- Largest files in /data")
get_file_sizes("data")
print("\n--- Largest files in /models")
get_file_sizes("models")
