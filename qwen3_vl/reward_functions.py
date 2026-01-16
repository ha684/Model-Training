# qwen3_vl/reward_functions.py
import json
import re

# --- Helper Functions (Giữ lại logic so sánh của bạn) ---
def check_exact_match(pred, gt):
    return 1.0 if str(pred).strip() == str(gt).strip() else 0.0

def check_inclusion(pred, gt):
    p, g = str(pred), str(gt)
    if g in p:
        # Phạt độ dài thừa: càng dài hơn ground truth càng bị trừ điểm
        len_diff = len(p) - len(g)
        return max(0.5 - (0.05 * len_diff), 0.0)
    return 0.0

def extract_json_from_text(text):
    """Cố gắng tìm và parse JSON từ output của model"""
    try:
        # Trường hợp model trả về JSON chuẩn ngay từ đầu
        return json.loads(text)
    except:
        pass
    
    # Trường hợp model bao JSON trong markdown ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
            
    # Trường hợp model quên markdown nhưng có { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
            
    return None

# --- Main Reward Functions (Đưa vào GRPO Trainer) ---

def format_reward_func(completions, **kwargs):
    """
    Reward function 1: Kiểm tra xem output có phải JSON hợp lệ không.
    Thưởng lớn nếu parse thành công để model học cách tuân thủ cú pháp.
    """
    rewards = []
    for completion in completions:
        data = extract_json_from_text(completion)
        if data is None:
            rewards.append(-1.0) # Phạt nặng nếu không ra JSON
        else:
            rewards.append(1.0) # Thưởng nếu đúng cú pháp JSON
    return rewards

def content_accuracy_reward_func(completions, ground_truth, **kwargs):
    """
    Reward function 2: So sánh nội dung chi tiết (Metadata & Table)
    Đây là nơi áp dụng logic Exact Match & Inclusion của bạn.
    """
    rewards = []
    
    # ground_truth trong TRL thường là list các string (raw json), cần parse lại
    # Lưu ý: dataset cần column "ground_truth" chứa json string hoặc dict
    
    for completion, gt_item in zip(completions, ground_truth):
        score = 0.0
        
        # 1. Parse Data
        pred_data = extract_json_from_text(completion)
        
        # Xử lý GT: nếu nó là string thì load ra, nếu là dict thì dùng luôn
        gt_data = gt_item
        if isinstance(gt_item, str):
            try:
                gt_data = json.loads(gt_item)
            except:
                rewards.append(0.0)
                continue

        if pred_data is None:
            rewards.append(-1.0) # Format sai thì khỏi chấm nội dung
            continue

        # 2. Chấm điểm METADATA
        # Lấy danh sách key từ GT để so sánh
        gt_meta = gt_data.get("metadata", {})
        pred_meta = pred_data.get("metadata", {})
        
        if not isinstance(pred_meta, dict):
            score -= 0.5
        else:
            for key, gt_val in gt_meta.items():
                pred_val = pred_meta.get(key)
                
                if gt_val is None: 
                    # Nếu GT là null, Pred cũng phải null hoặc "null"
                    if pred_val is None or str(pred_val).lower() == "null":
                        score += 0.2
                    else:
                        score -= 0.1
                    continue

                if pred_val is None:
                    score -= 0.2 # Mất dữ liệu
                    continue
                
                # Áp dụng logic Exact Match của bạn
                # Trọng số cao (2.0) vì đây là mục tiêu chính
                match_score = check_exact_match(pred_val, gt_val)
                score += match_score * 2.0 
                
                # Nếu không exact match, thử vớt vát bằng inclusion
                if match_score == 0:
                    score += check_inclusion(pred_val, gt_val)

        # 3. Chấm điểm TABLE ROWS
        gt_rows = gt_data.get("table_rows", [])
        pred_rows = pred_data.get("table_rows", [])
        
        # So sánh số lượng dòng
        if abs(len(gt_rows) - len(pred_rows)) > 0:
            score -= 0.5 * abs(len(gt_rows) - len(pred_rows)) # Phạt nếu sai số dòng
        
        for p_row, g_row in zip(pred_rows, gt_rows):
            if not isinstance(p_row, dict): continue
            
            for key, gt_val in g_row.items():
                pred_val = p_row.get(key)
                if gt_val is None: continue
                
                score += check_exact_match(pred_val, gt_val) * 1.5
        
        rewards.append(score)
        
    return rewards

def logic_constraint_reward_func(completions, **kwargs):
    """
    Reward function 3: Logic constraints (Regex date, invoice number...)
    """
    rewards = []
    for completion in completions:
        score = 0.0
        data = extract_json_from_text(completion)
        
        if data:
            meta = data.get("metadata", {})
            
            so_phieu = str(meta.get("Số phiếu", ""))
            if so_phieu and any(c.isalpha() for c in so_phieu):
                score -= 1.0
            
            date_str = str(meta.get("Ngày chứng từ(dd/mm/yyyy)", ""))
            if date_str and date_str.lower() != "none" and date_str != "null":
                if not re.match(r"^\d{2}/\d{2}/\d{4}$", date_str):
                    score -= 0.5
                    
        rewards.append(score)
    return rewards