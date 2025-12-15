# Senko Data Extraction Pipeline

Pipeline trích xuất dữ liệu từ PDF và ảnh (phiếu xuất kho, chứng từ) sử dụng Gemini Vision API.

## Tính năng

- ✅ Hỗ trợ PDF và ảnh (PNG, JPG, JPEG, WebP, GIF, BMP)
- ✅ Trích xuất dữ liệu tự động với Gemini Vision API
- ✅ Checkpoint/Resume: Lưu tiến trình, khởi động lại không mất dữ liệu
- ✅ API Key rotation: Tự động chuyển sang key khác khi hết quota
- ✅ Output JSON với append mode
- ✅ Dễ dàng thêm/bớt trường cần trích xuất

## Cài đặt

### 1. Cài đặt Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Cài đặt Poppler (required cho xử lý PDF)

**Windows:**
1. Download từ: https://github.com/oschwartz10612/poppler-windows/releases
2. Giải nén và thêm thư mục `bin/` vào PATH

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils
```

### 3. Cấu hình API Keys

Thêm Gemini API keys vào file `api_keys.txt` (mỗi key một dòng):

```
YOUR_GEMINI_API_KEY_1
YOUR_GEMINI_API_KEY_2
YOUR_GEMINI_API_KEY_3
```

Lấy API key tại: https://makersuite.google.com/app/apikey

## Sử dụng

### Chạy pipeline

```bash
python pipeline.py --input ./data --output ./output/results.json --api-keys ./api_keys.txt
```

### Các tham số

| Tham số | Viết tắt | Mô tả |
|---------|----------|-------|
| `--input` | `-i` | Thư mục hoặc file input (PDF, ảnh) |
| `--output` | `-o` | File JSON output |
| `--api-keys` | `-k` | File chứa Gemini API keys |
| `--checkpoint` | `-c` | File checkpoint (mặc định: cùng thư mục với output) |

### Ví dụ

```bash
# Xử lý thư mục documents
python pipeline.py -i ./documents -o ./extracted.json -k ./api_keys.txt

# Xử lý một file PDF cụ thể
python pipeline.py -i ./invoice.pdf -o ./invoice_data.json -k ./api_keys.txt

# Tiếp tục xử lý từ checkpoint
python pipeline.py -i ./data -o ./results.json -k ./api_keys.txt -c ./checkpoint.json
```

## Cấu trúc Output

```json
[
  {
    "source_file": "path/to/document.pdf",
    "source_type": "pdf",
    "page_number": 1,
    "total_pages": 3,
    "extracted_at": "2024-01-15T10:30:00",
    "metadata": [
      {
        "Số phiếu": "PXK-001",
        "Ngày chứng từ(mm/dd/yyyy)": "01/15/2024",
        "Ngày xuất(mm/dd/yyyy)": "01/15/2024",
        "Kho nhận": "Kho A",
        "Địa chỉ": "123 ABC Street",
        "Mã AR": "AR001"
      }
    ],
    "table_rows": [
      {
        "Mã FG": "FG001",
        "Đvt": "Thùng",
        "Số lượng(Thùng)": 10,
        "Số lượng(Pcs)": 100
      },
      {
        "Mã FG": "FG002",
        "Đvt": "Thùng",
        "Số lượng(Thùng)": 5,
        "Số lượng(Pcs)": 50
      }
    ]
  }
]
```

## Cấu hình Fields

Để thêm/bớt trường cần trích xuất, sửa file `config/fields_config.py`:

```python
# Tất cả các trường cần trích xuất
DEFAULT_FIELDS = [
    "Số phiếu",
    "Ngày chứng từ(mm/dd/yyyy)",
    # Thêm trường mới ở đây...
]

# Các trường thuộc metadata (thông tin cấp tài liệu)
METADATA_FIELDS = [
    "Số phiếu",
    # Thêm/bớt trường metadata ở đây...
]

# TABLE_ROW_FIELDS tự động = DEFAULT_FIELDS - METADATA_FIELDS
```

## Cấu trúc thư mục

```
senko_data/
├── config/
│   ├── __init__.py
│   └── fields_config.py    # Cấu hình fields và prompts
├── pipeline.py             # Pipeline chính
├── requirements.txt        # Dependencies
├── api_keys.txt           # API keys (không commit lên git!)
├── README.md
└── output/                # Thư mục output (tự tạo)
    ├── results.json       # Kết quả trích xuất
    └── checkpoint.json    # Checkpoint
```

## Lưu ý

1. **API Keys**: Không commit file `api_keys.txt` lên git. Thêm vào `.gitignore`.

2. **Checkpoint**: Pipeline tự động lưu checkpoint sau mỗi trang/ảnh. Nếu bị ngắt giữa chừng, chạy lại lệnh để tiếp tục.

3. **Rate Limiting**: Pipeline tự động delay giữa các request để tránh rate limit.

4. **API Key Rotation**: Khi một key hết quota, tự động chuyển sang key tiếp theo.

## License

MIT

