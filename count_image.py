import os

def count_jpg_files_recursive(path: str) -> int:
    if not os.path.isdir(path):
        raise ValueError(f"Path '{path}' không tồn tại hoặc không phải là thư mục.")

    count = 0
    for root, dirs, files in os.walk(path):
        count += sum(1 for f in files if f.lower().endswith(".jpg"))
    return count

if __name__ == "__main__":
    folder_path = input("Nhập đường dẫn thư mục: ").strip()
    try:
        total = count_jpg_files_recursive(folder_path)
        print(f"Tổng số file .jpg (kể cả thư mục con) trong '{folder_path}': {total}")
    except ValueError as e:
        print(e)
