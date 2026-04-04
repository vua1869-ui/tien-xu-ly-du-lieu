import os
import shutil

# Đường dẫn tới thư mục làm việc của bạn
base_path = r"c:\Users\lenk\OneDrive\Desktop\tienaxulydulieu"

# Từ khóa nhận diện file thuộc về Lab nào
lab_keywords = {
    "Lab1_Data": ["Lab_1", "Lab1"],
    "Lab2_Data": ["Lab_2", "Lab2"],
    "Lab3_Data": ["Lab_3", "Lab3"],
    "Lab4_Data": ["Lab_4", "Lab4"]
}

for folder_name, keywords in lab_keywords.items():
    folder_path = os.path.join(base_path, folder_name)
    
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Tìm và chuyển các file .csv, .pdf vào đúng thư mục
    for file in os.listdir(base_path):
        if file.endswith((".csv", ".pdf")):
            for kw in keywords:
                if kw.lower() in file.lower():
                    shutil.move(os.path.join(base_path, file), os.path.join(folder_path, file))
                    print(f"Đã di chuyển: {file} -> {folder_name}/")
                    break
                    
print("\n✅ Đã dọn dẹp xong! Thư mục của bạn giờ đã gọn gàng hơn.")