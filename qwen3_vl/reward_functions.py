# qwen3_vl/reward_functions.py

import json
import re

# ==============================================================================
# 1. HELPER FUNCTIONS (AUTO-DETECTION)
# ==============================================================================

def extract_json_from_text(text):
    """
    Tìm và parse JSON từ text bất kỳ.
    Ưu tiên: Markdown block -> Object {} -> List [] -> Raw Text
    """
    text = text.strip()
    
    # 1. Thử parse trực tiếp (nếu model chỉ trả về đúng chuỗi JSON sạch)
    try:
        return json.loads(text)
    except:
        pass
    
    # 2. Tìm block markdown ```json ... ```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
            
    # 3. Tìm cấu trúc object { ... } (Greedy match: lấy cái to nhất)
    # Tìm dấu { đầu tiên và dấu } cuối cùng
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try: return json.loads(text[start:end+1])
        except: pass
            
    # 4. Tìm cấu trúc list [ ... ] (Nếu output là list)
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try: return json.loads(text[start:end+1])
        except: pass
            
    return None

def normalize_value(val):
    """Chuẩn hóa để so sánh"""
    if val is None: return None
    s = str(val).strip()
    if s.lower() in ["null", "none", "nan", ""]: return None
    return s

def check_exact_match(pred, gt):
    return 1.0 if normalize_value(pred) == normalize_value(gt) else 0.0

def check_inclusion(pred, gt):
    p, g = normalize_value(pred), normalize_value(gt)
    if p is None or g is None: return 0.0
    if g in p:
        len_diff = len(p) - len(g)
        return max(0.5 - (0.05 * len_diff), 0.0)
    return 0.0

# ==============================================================================
# 2. REWARD FUNCTIONS (NO TAGS MODE)
# ==============================================================================

def format_reward_func(completions, **kwargs):
    """
    Reward 1: JSON Syntax Check.
    Mục tiêu: Output phải chứa JSON hợp lệ, bất kể nằm ở đâu.
    """
    rewards = []
    for completion in completions:
        data = extract_json_from_text(completion)
        if data is None:
            # Phạt rất nặng vì đây là mục tiêu tối thượng
            rewards.append(-2.0) 
        else:
            # Thưởng vì đã sinh ra được JSON parse được
            rewards.append(1.0) 
    return rewards

def content_accuracy_reward_func(completions, ground_truth, **kwargs):
    """
    Reward 2: Semantic Accuracy Check.
    So khớp dữ liệu field-by-field.
    """
    rewards = []
    
    for completion, gt_item in zip(completions, ground_truth):
        score = 0.0
        
        # --- Parse GT ---
        gt_data = gt_item
        if isinstance(gt_item, str):
            try: gt_data = json.loads(gt_item)
            except: rewards.append(0.0); continue
            
        # Unwrap list logic (cho dataset của bạn)
        if isinstance(gt_data, list) and len(gt_data) > 0: gt_data = gt_data[0]
        if not isinstance(gt_data, dict): gt_data = {}

        # --- Parse Pred ---
        # Không cần extract thẻ nữa, parse thẳng từ completion
        pred_data = extract_json_from_text(completion)
        
        if pred_data is None:
            rewards.append(-1.0); continue

        # Unwrap list logic (cho prediction)
        if isinstance(pred_data, list) and len(pred_data) > 0: pred_data = pred_data[0]
        if not isinstance(pred_data, dict): 
            rewards.append(-1.0); continue

        # --- Compare Metadata ---
        gt_meta = gt_data.get("metadata", {})
        pred_meta = pred_data.get("metadata", {})
        
        # Fix format
        if isinstance(gt_meta, list) and len(gt_meta) > 0: gt_meta = gt_meta[0]
        if isinstance(pred_meta, list) and len(pred_meta) > 0: pred_meta = pred_meta[0]
        
        if isinstance(gt_meta, dict) and isinstance(pred_meta, dict):
            for k, g_v in gt_meta.items():
                p_v = pred_meta.get(k)
                if normalize_value(g_v) is None:
                    if normalize_value(p_v) is None: score += 0.2
                    else: score -= 0.1
                else:
                    if check_exact_match(p_v, g_v): score += 2.0
                    elif check_inclusion(p_v, g_v): score += 1.0

        # --- Compare Table Rows ---
        gt_rows = gt_data.get("table_rows", [])
        pred_rows = pred_data.get("table_rows", [])
        
        if not isinstance(gt_rows, list): gt_rows = []
        if not isinstance(pred_rows, list): pred_rows = []

        # Phạt lệch dòng
        score -= (abs(len(gt_rows) - len(pred_rows)) * 0.5)

        # So khớp dòng
        min_len = min(len(gt_rows), len(pred_rows))
        for i in range(min_len):
            g_row = gt_rows[i]
            p_row = pred_rows[i]
            
            # Ưu tiên check Mã trước
            if check_exact_match(p_row.get("Mã FG"), g_row.get("Mã FG")):
                score += 3.0 
            
            # Check Số lượng
            if check_exact_match(p_row.get("Số lượng"), g_row.get("Số lượng")):
                score += 2.0
            
            # Check ĐVT (Optional - thêm vào nếu cần)
            if check_exact_match(p_row.get("Đvt"), g_row.get("Đvt")):
                score += 1.0

        rewards.append(score)
        
    return rewards