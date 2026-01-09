import json

# Giả lập chuỗi JSON "xấu xí" bạn vừa nhận được
messy_json_str = """
{
    "assistant": "{\\\"nha_cung_cap\\\": {\\\"ten_cong_ty\\\": \\\"KOIKEYA...\\\"}}"
}
"""

# Bước 1: Load file tổng (Level 1)
data = json.loads(messy_json_str)
assistant_response = data["assistant"]

print("--- Level 1: Lấy nội dung Assistant (Dạng chuỗi) ---")
print(assistant_response) 
# Kết quả vẫn sẽ có dấu \ nhưng bớt đi 1 lớp

# Bước 2: Load nội dung Assistant (Level 2)
# Đây là bước các ứng dụng sau này sẽ làm để lấy dữ liệu sạch
try:
    inner_data = json.loads(assistant_response)
    print("\n--- Level 2: Dữ liệu sạch (Dạng Object) ---")
    print(json.dumps(inner_data, indent=2, ensure_ascii=False))
except:
    print("Đây là text thường, không phải JSON")