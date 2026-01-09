import json
import os

# Cấu hình đường dẫn file
INPUT_FILE = './output/conversations.json'  # Đường dẫn file hiện tại của bạn
OUTPUT_FILE = './output/conversations.json' # File mới đã sửa

def fix_dataset(input_path, output_path):
    print(f"Đang đọc file: {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Không tìm thấy file input!")
        return

    fixed_count = 0
    
    # Duyệt qua từng đoạn hội thoại
    for entry in data:
        if "conversation" in entry:
            for turn in entry["conversation"]:
                assistant_content = turn.get("assistant")
                
                # KIỂM TRA QUAN TRỌNG:
                # Nếu nó là Dict hoặc List -> Chuyển thành String (Dump nó ra)
                if isinstance(assistant_content, (dict, list)):
                    turn["assistant"] = json.dumps(assistant_content, ensure_ascii=False)
                    fixed_count += 1
                # Nếu là None -> Chuyển thành chuỗi rỗng hoặc xử lý tùy ý
                elif assistant_content is None:
                    turn["assistant"] = ""

    print(f"Đã sửa {fixed_count} lượt hội thoại từ Object -> String.")

    # Lưu file mới
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Đã lưu file sửa xong tại: {output_path}")
    print("Bạn hãy kiểm tra file mới, phần assistant sẽ có nhiều dấu gạch chéo (\) là ĐÚNG.")

if __name__ == "__main__":
    fix_dataset(INPUT_FILE, OUTPUT_FILE)