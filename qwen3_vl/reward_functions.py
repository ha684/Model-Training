import json
import re
from datetime import datetime

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def extract_content_from_solution_tag(text):
    """
    Trích xuất nội dung giữa thẻ <SOLUTION> và </SOLUTION>.
    Nếu không tìm thấy thẻ, trả về text gốc (fallback).
    """
    pattern = r"<SOLUTION>(.*?)</SOLUTION>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

def extract_json_from_text(text):
    """
    Cố gắng parse JSON từ text, hỗ trợ cả markdown block.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    
    # Tìm block ```json ... ```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
            
    # Tìm cấu trúc { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
            
    return None

def normalize_value(val):
    """Chuẩn hóa giá trị để so sánh (đưa về string, strip space, lower null)"""
    if val is None: return None
    s = str(val).strip()
    if s.lower() in ["null", "none", "nan", ""]:
        return None
    return s

def check_exact_match(pred, gt):
    p = normalize_value(pred)
    g = normalize_value(gt)
    if p == g: return 1.0
    return 0.0

def check_inclusion(pred, gt):
    p = normalize_value(pred)
    g = normalize_value(gt)
    if p is None or g is None: return 0.0
    
    # Kiểm tra xem GT có nằm trong Pred không (chấp nhận model nói dài dòng)
    if g in p:
        len_diff = len(p) - len(g)
        # Điểm tối đa 0.5, trừ dần nếu nói quá thừa thãi
        return max(0.5 - (0.05 * len_diff), 0.0)
    return 0.0

# ==============================================================================
# 2. REWARD FUNCTIONS
# ==============================================================================

def xml_structure_reward_func(completions, **kwargs):
    """
    Reward 1: Kiểm tra cấu trúc thẻ XML.
    Mục tiêu: Bắt buộc model phải suy luận trước khi trả lời.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        
        # Check Reasoning tag
        has_reasoning = "<REASONING>" in completion and "</REASONING>" in completion
        if has_reasoning:
            score += 0.5
        
        # Check Solution tag
        has_solution = "<SOLUTION>" in completion and "</SOLUTION>" in completion
        if has_solution:
            score += 0.5
            
        # Check thứ tự: Reasoning phải nằm trước Solution
        if has_reasoning and has_solution:
            r_end = completion.find("</REASONING>")
            s_start = completion.find("<SOLUTION>")
            if r_end < s_start:
                score += 0.5
            else:
                score -= 0.5 # Trừ điểm nếu thứ tự lộn xộn
        
        rewards.append(score)
    return rewards

def format_reward_func(completions, **kwargs):
    """
    Reward 2: Kiểm tra cú pháp JSON cơ bản.
    Mục tiêu: Đảm bảo output trong thẻ SOLUTION là JSON hợp lệ.
    """
    rewards = []
    for completion in completions:
        json_text = extract_content_from_solution_tag(completion)
        data = extract_json_from_text(json_text)
        if data is None:
            rewards.append(-1.0) # Phạt nặng lỗi syntax
        else:
            rewards.append(1.0) # Thưởng vì đúng cú pháp
    return rewards

def content_accuracy_reward_func(completions, ground_truth, **kwargs):
    """
    Reward 3: Chấm điểm độ chính xác nội dung (Semantic check).
    FIXED: Handles cases where JSON is a List instead of a Dict.
    """
    rewards = []
    
    for completion, gt_item in zip(completions, ground_truth):
        score = 0.0
        
        # --- 1. Parse Ground Truth (GT) ---
        gt_data = gt_item
        if isinstance(gt_item, str):
            try: 
                gt_data = json.loads(gt_item)
            except: 
                rewards.append(0.0)
                continue

        # FIX: Handle if GT is a list (e.g., [{"metadata":...}])
        if isinstance(gt_data, list):
            gt_data = gt_data[0] if len(gt_data) > 0 else {}
            
        # Safety check: if it's still not a dict, default to empty to avoid crash
        if not isinstance(gt_data, dict):
            gt_data = {}

        # --- 2. Parse Prediction (Pred) ---
        json_text = extract_content_from_solution_tag(completion)
        pred_data = extract_json_from_text(json_text)

        if pred_data is None:
            rewards.append(-1.0)
            continue
            
        # FIX: Handle if Pred is a list
        if isinstance(pred_data, list):
            pred_data = pred_data[0] if len(pred_data) > 0 else {}

        if not isinstance(pred_data, dict):
            # If prediction isn't a dict (and wasn't a list of dicts), it's bad structure
            rewards.append(-1.0)
            continue

        # --- 3. Compare Metadata ---
        # Now safe to use .get() because we ensured they are dicts
        gt_meta = gt_data.get("metadata", {})
        pred_meta = pred_data.get("metadata", {})

        # Fix List/Dict nesting inside metadata keys (Defensive coding)
        if isinstance(gt_meta, list): 
            gt_meta = gt_meta[0] if gt_meta else {}
        if isinstance(pred_meta, list): 
            pred_meta = pred_meta[0] if pred_meta else {}

        if not isinstance(pred_meta, dict):
            score -= 1.0
        else:
            # Ensure gt_meta is iterable before .items()
            if not isinstance(gt_meta, dict):
                gt_meta = {}
                
            for key, gt_val in gt_meta.items():
                pred_val = pred_meta.get(key)
                
                # Check Null
                if normalize_value(gt_val) is None:
                    if normalize_value(pred_val) is None: score += 0.5
                    else: score -= 0.2
                    continue

                if normalize_value(pred_val) is None:
                    score -= 0.5; continue
                
                # Check Match
                match_score = check_exact_match(pred_val, gt_val)
                if match_score > 0:
                    score += 2.0
                else:
                    score += check_inclusion(pred_val, gt_val)

        # --- 4. Compare Table Rows ---
        gt_rows = gt_data.get("table_rows", [])
        pred_rows = pred_data.get("table_rows", [])
        
        if not isinstance(gt_rows, list): gt_rows = []
        if not isinstance(pred_rows, list): pred_rows = []

        # Phạt lệch số dòng
        if len(gt_rows) != len(pred_rows):
            score -= 0.5 * abs(len(gt_rows) - len(pred_rows))
        
        # So sánh từng dòng
        min_len = min(len(gt_rows), len(pred_rows))
        for i in range(min_len):
            g_row = gt_rows[i]
            p_row = pred_rows[i]
            if not isinstance(p_row, dict) or not isinstance(g_row, dict): continue
            
            for k, g_v in g_row.items():
                p_v = p_row.get(k)
                if normalize_value(g_v) is None: continue
                
                match_s = check_exact_match(p_v, g_v)
                score += match_s * 1.5
                if match_s == 0: 
                    score += check_inclusion(p_v, g_v)
        
        rewards.append(score)
        
    return rewards

def logic_constraint_reward_func(completions, **kwargs):
    """
    Reward 4: Kiểm tra Logic nghiệp vụ (Ngày tháng, định dạng số).
    Mục tiêu: Model không được hallucinate ra định dạng ngày lạ.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        json_text = extract_content_from_solution_tag(completion)
        data = extract_json_from_text(json_text)
        
        if data:
            meta = data.get("metadata", {})
            if isinstance(meta, list) and meta: meta = meta[0]
            
            # Check ngày tháng (Ví dụ: dd/mm/yyyy)
            # Bạn có thể thay đổi key này theo data thực tế của bạn
            date_keys = [k for k in meta.keys() if "ngày" in k.lower() or "date" in k.lower()]
            
            for k in date_keys:
                val = str(meta[k])
                if val.lower() not in ["none", "null", "nan", ""]:
                    # Regex check dd/mm/yyyy
                    if not re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", val):
                        score -= 0.5 # Phạt nếu sai format ngày
            
            # Check Số phiếu (không được chứa ký tự đặc biệt lạ nếu cần)
            so_phieu = str(meta.get("Số phiếu", ""))
            if so_phieu and len(so_phieu) > 20: # Ví dụ: quá dài là sai
                 score -= 0.5

        rewards.append(score)
    return rewards