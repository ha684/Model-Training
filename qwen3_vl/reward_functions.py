import json
import re
from difflib import SequenceMatcher

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def extract_json_from_text(text):
    text = text.strip()
    try: return json.loads(text)
    except: pass
    
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
            
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try: return json.loads(text[start:end+1])
        except: pass
        
    return None

def normalize_text(val):
    if val is None: return None
    s = str(val).strip()
    if s.lower() in ["null", "none", "nan", ""]: return None
    return s

def normalize_number_string(val):
    if val is None: return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(val))

def similarity_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def smart_compare(key, pred_val, gt_val):
    p = normalize_text(pred_val)
    g = normalize_text(gt_val)
    if p is None and g is None: return 1.0
    if p is None or g is None: return 0.0

    key_lower = key.lower()

    if "ngày" in key_lower or "date" in key_lower:
        return 1.0 if p == g else 0.0

    if "số phiếu" in key_lower or "phone" in key_lower or "fax" in key_lower:
        p_clean = normalize_number_string(p)
        g_clean = normalize_number_string(g)
        return 1.0 if p_clean == g_clean else 0.0

    if "mã ar" in key_lower or "mã kh" in key_lower:
        if p == g: return 1.0
        if (len(p) > 3 and p in g) or (len(g) > 3 and g in p): return 1.0
        return 0.0

    if "địa chỉ" in key_lower or "kho" in key_lower or "tên" in key_lower:
        if p == g: return 1.0
        sim = similarity_ratio(p.lower(), g.lower())
        if sim > 0.85: return sim 
        return 0.0

    return 1.0 if p == g else 0.0

# ==============================================================================
# 2. REWARD FUNCTIONS (UPDATED TABLE LOGIC)
# ==============================================================================

def format_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        score = 0.0
        if len(completion) > 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0
        
        data = extract_json_from_text(completion)
        if data is None:
            score -= 2.0
        else:
            score += 1.0
        rewards.append(score)
    return rewards

def content_accuracy_reward_func(completions, answer, **kwargs):
    """
    Check Content Accuracy với logic Unordered Table Rows
    """
    rewards = []
    ground_truth = answer
    for completion, gt_item in zip(completions, ground_truth):
        score = 0.0
        
        # --- Parse GT ---
        gt_data = gt_item
        if isinstance(gt_item, str):
            try: gt_data = json.loads(gt_item)
            except: rewards.append(0.0); continue
        
        if isinstance(gt_data, list) and len(gt_data) > 0: gt_data = gt_data[0]
        if not isinstance(gt_data, dict): gt_data = {}

        # --- Parse Pred ---
        pred_data = extract_json_from_text(completion)
        if pred_data is None:
            rewards.append(-1.0); continue
            
        if isinstance(pred_data, list) and len(pred_data) > 0: pred_data = pred_data[0]
        if not isinstance(pred_data, dict): 
            rewards.append(-1.0); continue

        # ----------------------------------------------------------------------
        # 1. COMPARE METADATA (Giữ nguyên logic cũ)
        # ----------------------------------------------------------------------
        gt_meta = gt_data.get("metadata", {})
        pred_meta = pred_data.get("metadata", {})
        
        if isinstance(gt_meta, list) and len(gt_meta) > 0: gt_meta = gt_meta[0]
        if isinstance(pred_meta, list) and len(pred_meta) > 0: pred_meta = pred_meta[0]

        if isinstance(gt_meta, dict) and isinstance(pred_meta, dict):
            for k, g_v in gt_meta.items():
                p_v = pred_meta.get(k)
                match_score = smart_compare(k, p_v, g_v)
                if match_score == 1.0: score += 2.0
                elif match_score > 0.8: score += 1.5
                else:
                    if normalize_text(g_v) is None and normalize_text(p_v) is None: score += 0.5
                    elif normalize_text(p_v) is not None: score -= 0.5

        # ----------------------------------------------------------------------
        # 2. COMPARE TABLE ROWS (UPDATED: UNORDERED MATCHING)
        # ----------------------------------------------------------------------
        gt_rows = gt_data.get("table_rows", [])
        pred_rows = pred_data.get("table_rows", [])
        
        if not isinstance(gt_rows, list): gt_rows = []
        if not isinstance(pred_rows, list): pred_rows = []

        # Tạo một bản sao (copy) danh sách pred_rows để quản lý các dòng đã match
        # Mục đích: Mỗi dòng pred chỉ được match với 1 dòng GT
        available_preds = pred_rows.copy()

        # Phạt lệch số lượng dòng
        score -= (abs(len(gt_rows) - len(pred_rows)) * 0.5)

        for g_row in gt_rows:
            best_match_idx = -1
            best_match_score = 0.0 # Điểm match tối thiểu để chấp nhận

            # Tìm dòng tương ứng trong available_preds
            # Key chính để tìm là "Mã FG"
            g_fg = normalize_text(g_row.get("Mã FG"))
            
            # --- Chiến thuật tìm kiếm ---
            # Duyệt qua các dòng Pred còn lại để tìm dòng khớp Mã FG
            found_candidate_indices = []
            
            for idx, p_row in enumerate(available_preds):
                if not isinstance(p_row, dict): continue
                p_fg = normalize_text(p_row.get("Mã FG"))
                
                # Check Mã FG trước
                if p_fg == g_fg and g_fg is not None:
                    found_candidate_indices.append(idx)
            
            # Nếu tìm thấy các dòng khớp Mã FG, chọn dòng có Số lượng khớp nhất (nếu có duplicate mã)
            target_idx = -1
            
            if len(found_candidate_indices) > 0:
                # Nếu chỉ có 1 dòng khớp mã -> Lấy luôn
                if len(found_candidate_indices) == 1:
                    target_idx = found_candidate_indices[0]
                else:
                    # Nếu có nhiều dòng trùng mã FG, so tiếp Số lượng để chọn dòng đúng nhất
                    g_qty = normalize_number_string(g_row.get("Số lượng"))
                    for idx in found_candidate_indices:
                        p_qty = normalize_number_string(available_preds[idx].get("Số lượng"))
                        if p_qty == g_qty:
                            target_idx = idx
                            break
                    # Nếu vẫn không khớp số lượng, lấy thằng đầu tiên trùng mã
                    if target_idx == -1: target_idx = found_candidate_indices[0]
            
            # --- Chấm điểm dựa trên dòng tìm được ---
            if target_idx != -1:
                # Tìm thấy cặp tương ứng -> Chấm điểm
                p_row = available_preds[target_idx]
                
                # Điểm cho Mã FG (chắc chắn đúng mới vào đây)
                score += 3.0 
                
                # Điểm cho Số lượng
                sl_score = smart_compare("Số lượng", p_row.get("Số lượng"), g_row.get("Số lượng"))
                if sl_score == 1.0: score += 2.0
                
                # Điểm cho ĐVT
                dvt_score = smart_compare("Đvt", p_row.get("Đvt"), g_row.get("Đvt"))
                if dvt_score > 0.8: score += 1.0

                # Xóa dòng đã dùng khỏi danh sách để không match lại lần sau
                available_preds.pop(target_idx)
            
            else:
                # Không tìm thấy dòng nào có Mã FG khớp -> Không có điểm dòng này
                # Có thể thêm phạt nhẹ nếu muốn strict hơn
                pass

        rewards.append(score)
        
    return rewards