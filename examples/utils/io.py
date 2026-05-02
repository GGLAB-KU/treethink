import json
import os


def get_relative_path(start_directory, target_directory):
    """Başlangıç dizininden hedef dizine göreceli yolu döndürür."""
    # Başlangıç ve hedef dizinlerin mutlak yollarını al
    start_abs = os.path.abspath(start_directory)
    target_abs = os.path.abspath(target_directory)
    # Göreceli yolu hesapla
    relative_path = os.path.relpath(target_abs, start_abs)
    return relative_path


def list_directories(directory):
    """List all directories in the given directory, excluding any .lock files."""
    try:
        # List all entries in the specified path and filter directories, ignoring .lock files
        all_entries = os.listdir(directory)
        directories = [
            entry
            for entry in all_entries
            if os.path.isdir(os.path.join(directory, entry))
            and not entry.endswith(".locks")
        ]
        # replace -- with /
        directories = [entry.replace("--", "/") for entry in directories]
        # replace models/ with ''
        directories = [entry.replace("models/", "") for entry in directories]

        return directories
    except Exception as e:
        print(f"Error: {e}")
        return []


# file_path='user_input.json'
def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data
