import json
import re

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
    
    if g in p:
        # Logic cũ: Phạt nếu string dự đoán quá dài so với đáp án
        len_diff = len(p) - len(g)
        return max(0.5 - (0.05 * len_diff), 0.0)
    return 0.0

# --- 2. Main Reward Function ---

def content_accuracy_reward_func(completions, ground_truth, **kwargs):
    """
    Reward function chấm điểm độ chính xác nội dung JSON.
    Hỗ trợ format: <REASONING>...</REASONING> <SOLUTION>JSON</SOLUTION>
    """
    rewards = []
    
    for completion, gt_item in zip(completions, ground_truth):
        score = 0.0
        
        # --- BƯỚC 1: Lấy Ground Truth ---
        gt_data = gt_item
        if isinstance(gt_item, str):
            try:
                gt_data = json.loads(gt_item)
            except:
                # Nếu GT lỗi thì không thể chấm, trả về 0 (hoặc skip)
                rewards.append(0.0)
                continue

        # --- BƯỚC 2: Lấy Prediction ---
        # Ưu tiên lấy trong thẻ <SOLUTION>, nếu không có thì lấy raw
        json_text = extract_content_from_solution_tag(completion)
        pred_data = extract_json_from_text(json_text)

        if pred_data is None:
            rewards.append(-1.0) # Phạt nặng nếu không ra được JSON
            continue

        # --- BƯỚC 3: Chấm METADATA ---
        # Xử lý linh hoạt: metadata có thể là dict hoặc list chứa 1 dict
        gt_meta = gt_data.get("metadata", {})
        pred_meta = pred_data.get("metadata", {})

        # Chuẩn hóa về Dict: Nếu là list thì lấy phần tử đầu tiên
        if isinstance(gt_meta, list):
            gt_meta = gt_meta[0] if len(gt_meta) > 0 else {}
        if isinstance(pred_meta, list):
            pred_meta = pred_meta[0] if len(pred_meta) > 0 else {}

        if not isinstance(pred_meta, dict):
            score -= 1.0 # Sai cấu trúc metadata
        else:
            # Duyệt qua từng key trong Ground Truth để chấm
            for key, gt_val in gt_meta.items():
                pred_val = pred_meta.get(key)
                
                # Case 1: GT là Null
                if normalize_value(gt_val) is None:
                    if normalize_value(pred_val) is None:
                        score += 0.5 # Đúng là Null
                    else:
                        score -= 0.2 # GT Null mà Pred có giá trị -> Hallucination
                    continue

                # Case 2: Pred bị thiếu key hoặc Null
                if normalize_value(pred_val) is None:
                    score -= 0.5 # Bỏ sót thông tin quan trọng
                    continue
                
                # Case 3: So sánh giá trị
                match_score = check_exact_match(pred_val, gt_val)
                if match_score > 0:
                    score += 2.0 # Thưởng lớn cho Exact Match
                else:
                    # Nếu không exact match, thử Inclusion (vớt vát điểm)
                    score += check_inclusion(pred_val, gt_val)

        # --- BƯỚC 4: Chấm TABLE ROWS ---
        gt_rows = gt_data.get("table_rows", [])
        pred_rows = pred_data.get("table_rows", [])
        
        # Đảm bảo là list
        if not isinstance(gt_rows, list): gt_rows = []
        if not isinstance(pred_rows, list): pred_rows = []

        # Phạt lệch số dòng
        len_diff = abs(len(gt_rows) - len(pred_rows))
        if len_diff > 0:
            score -= 0.5 * len_diff
        
        # So sánh từng dòng (theo thứ tự)
        # Lưu ý: Logic này giả định thứ tự dòng phải khớp. 
        # Nếu bài toán không quan trọng thứ tự, cần logic so khớp phức tạp hơn.
        min_len = min(len(gt_rows), len(pred_rows))
        for i in range(min_len):
            g_row = gt_rows[i]
            p_row = pred_rows[i]
            
            if not isinstance(p_row, dict) or not isinstance(g_row, dict):
                continue
                
            for k, g_v in g_row.items():
                p_v = p_row.get(k)
                
                # Logic chấm cell tương tự metadata nhưng trọng số thấp hơn chút
                if normalize_value(g_v) is None:
                    continue # Bỏ qua check null trong bảng để đỡ gắt
                
                if normalize_value(p_v) is None:
                    score -= 0.1
                    continue
                    
                match_s = check_exact_match(p_v, g_v)
                score += match_s * 1.5 # Thưởng 1.5 cho mỗi cell đúng
        
        rewards.append(score)
        
    return rewards