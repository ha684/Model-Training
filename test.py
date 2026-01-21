import io
import sys

from qwen3_vl.evaluation import detect_language
from qwen3_vl.utils import get_system_and_user_prompt

# Set stdout to UTF-8 for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


gt_data = {"metadata": [{"document_number": None, "receiving_warehouse": "BIG C THANH HÓA", "address": "Xã Đông Hải, Thành Phố Thanh Hóa, Tỉnh Thanh Hóa", "ar_code": "AR1831___141", "document_date": "08/05/2025", "export_date": None}], "table_rows": [{"product_code": "FG2351", "unit": "Cái", "quantity": 144.0}]}

language = detect_language(gt_data)
system_prompt, user_prompt = get_system_and_user_prompt(language, sample=gt_data)

print(f"Language: {language}")
print(f"System prompt: {system_prompt}")
print(f"User prompt: {user_prompt}")