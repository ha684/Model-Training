# config/fields_config_vn.py
# -*- coding: utf-8 -*-
"""
Configuration file for data extraction fields and prompts.
Modify the fields lists below to add/remove fields as needed.
"""

DEFAULT_FIELDS_VI = [
    "Số phiếu",
    "Ngày chứng từ(dd/mm/yyyy)",
    "Ngày xuất(dd/mm/yyyy)",
    "Mã FG",
    "Đvt",
    "Số lượng",
    "Kho nhận",
    "Địa chỉ",
    "Mã AR",
]

METADATA_FIELDS_VI = [
    "Số phiếu",
    "Kho nhận",
    "Địa chỉ",
    "Mã AR",
    "Ngày chứng từ(dd/mm/yyyy)",
    "Ngày xuất(dd/mm/yyyy)",
]

TABLE_ROW_FIELDS_VI = [f for f in DEFAULT_FIELDS_VI if f not in METADATA_FIELDS_VI]


def get_extraction_prompt_vi() -> str:
    """Tạo prompt trích xuất bằng tiếng Việt với tên field tiếng Việt."""
    
    prompt = f"""Trích xuất dữ liệu có cấu trúc từ phiếu xuất kho tiếng Việt này và trả về dưới dạng JSON.

YÊU CẦU ĐẦU RA:
- Trả về dữ liệu dưới định dạng JSON với cấu trúc sau:
{{
  "metadata": {{
    {", ".join([f'"{f}": <giá trị hoặc null>' for f in METADATA_FIELDS_VI])}
  }},
  "table_rows": [
    {{
      {", ".join([f'"{f}": <giá trị hoặc null>' for f in TABLE_ROW_FIELDS_VI])}
    }}
  ]
}}

#HƯỚNG DẪN TRÍCH XUẤT:

### CÁC TRƯỜNG METADATA:

1. Số phiếu
   - Số chứng từ/PO chính, thường ở góc trên bên phải
   - Các định dạng phổ biến: "PXB25110076", "2544051804257", "252.104.868.0263"
   - Trích xuất chính xác như hiển thị, giữ nguyên dấu chấm, gạch ngang, khoảng trắng
   - KHÔNG phải số trang viết tay như "1/2" hoặc số sau mã khách hàng

2. Ngày chứng từ(dd/mm/yyyy)
   - Ngày lập phiếu theo định dạng NGÀY/THÁNG/NĂM (dd/mm/yyyy)
   - VÍ DỤ: "15/03/2025" nghĩa là ngày 15 tháng 3 năm 2025
   - QUAN TRỌNG: Format Việt Nam là NGÀY trước, THÁNG sau (dd/mm/yyyy), KHÔNG PHẢI mm/dd/yyyy
   - Trích xuất chính xác như hiển thị trên chứng từ

3. Ngày xuất(dd/mm/yyyy)
   - Ngày xuất kho hoặc ngày giao hàng theo định dạng dd/mm/yyyy
   - Trích xuất chính xác như hiển thị trên chứng từ
   - Có thể trùng hoặc khác với Ngày chứng từ

4. Kho nhận
   - Tên kho hoặc khách hàng nhận hàng
   - Ví dụ: "DC KHO HẢI PHÒNG", "Kho Hà Nội"
   - Trích xuất tên đầy đủ như hiển thị

5. Địa chỉ
   - Địa chỉ giao hàng đầy đủ
   - Trường tùy chọn - đặt null nếu không có trên chứng từ
   - Không đoán hoặc suy luận

6. Mã AR
   - Mã tài khoản khách hàng
   - Ví dụ: "A68/750", "A68/750_141", "AR1830"
   - Trích xuất mã đầy đủ bao gồm cả hậu tố sau dấu gạch dưới nếu có

### CÁC TRƯỜNG TRONG BẢNG:

7. Mã FG
   - Mã sản phẩm, thường là cột ngoài cùng bên trái trong bảng
   - Ví dụ: "PG1490", "FC9345", "PL2368"
   - Trích xuất chính xác mã chữ và số

8. Đvt
   - Đơn vị tính của sản phẩm
   - Ví dụ: "Hộp", "Gói", "Chai", "Thùng"
   - Giữ nguyên văn bản

9. Số lượng
   - Giá trị số lượng dạng số
   - QUAN TRỌNG: Phải là kiểu số (number), không phải chuỗi (string)
   - Ví dụ: {{"Số lượng": 168}} KHÔNG PHẢI {{"Số lượng": "168"}}

QUY TẮC QUAN TRỌNG:
- CHỈ trả về JSON hợp lệ, không có markdown formatting hay giải thích
- Trường Số lượng BẮT BUỘC là kiểu số (integer hoặc float), các trường khác là chuỗi hoặc null
- Ngày đã ở định dạng dd/mm/yyyy - trích xuất chính xác như hiển thị
- Mỗi dòng sản phẩm trong bảng tạo một object trong mảng table_rows
- Giữ nguyên dấu tiếng Việt trong giá trị văn bản (ă, â, đ, ê, ô, ơ, ư, v.v.)
- Đặt trường là null nếu không tìm thấy - không đoán hoặc giả định
- Số phiếu ở phần header, KHÔNG phải số xuất hiện sau mã khách hàng
- Bao gồm Mã AR đầy đủ với hậu tố nếu có
"""
    
    return prompt