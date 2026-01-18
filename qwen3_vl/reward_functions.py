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
    rewards = []
    ground_truth = answer
    
    for completion, gt_item in zip(completions, ground_truth):
        score = 0.0
        
        # --- 1. Parse Data (Giữ nguyên logic cũ) ---
        gt_data = gt_item
        if isinstance(gt_item, str):
            try: gt_data = json.loads(gt_item)
            except: rewards.append(-5.0); continue # Phạt rất nặng nếu GT lỗi (hiếm) hoặc logic sai
            
        if isinstance(gt_data, list) and len(gt_data) > 0: gt_data = gt_data[0]
        if not isinstance(gt_data, dict): gt_data = {}

        pred_data = extract_json_from_text(completion)
        if pred_data is None:
            rewards.append(-5.0); continue # Phạt nặng nếu không ra JSON
            
        if isinstance(pred_data, list) and len(pred_data) > 0: pred_data = pred_data[0]
        if not isinstance(pred_data, dict): 
            rewards.append(-5.0); continue

        # ======================================================================
        # 2. METADATA SCORING (STRICT MODE)
        # ======================================================================
        gt_meta = gt_data.get("metadata", {})
        pred_meta = pred_data.get("metadata", {})
        
        # Flatten list nếu cần
        if isinstance(gt_meta, list): gt_meta = gt_meta[0] if gt_meta else {}
        if isinstance(pred_meta, list): pred_meta = pred_meta[0] if pred_meta else {}

        # Duyệt qua tất cả keys trong GT Metadata
        for k, g_v in gt_meta.items():
            p_v = pred_meta.get(k)
            
            # Logic so sánh
            match_score = smart_compare(k, p_v, g_v)
            
            if match_score == 1.0:
                score += 2.0  # Thưởng: Extract chuẩn tuyệt đối
            elif match_score > 0.85:
                score += 1.0  # Thưởng: Extract chấp nhận được (Fuzzy)
            else:
                # PHẠT: Nếu field này có trong GT mà model extract sai hoặc null
                # Chỉ phạt nếu GT có dữ liệu thực sự
                if normalize_text(g_v) is not None:
                     score -= 2.0 # Phạt nặng vì sai thông tin Metadata
        
        # Phạt hallucination: Model bịa ra field không có trong GT
        for k, p_v in pred_meta.items():
            if k not in gt_meta and normalize_text(p_v) is not None:
                score -= 1.0 # Phạt nhẹ tội bịa field thừa

        # ======================================================================
        # 3. TABLE ROWS SCORING (HIGH PENALTY MODE)
        # ======================================================================
        gt_rows = gt_data.get("table_rows", [])
        pred_rows = pred_data.get("table_rows", [])
        
        if not isinstance(gt_rows, list): gt_rows = []
        if not isinstance(pred_rows, list): pred_rows = []

        # Phạt chênh lệch số dòng (Missed Detection hoặc Hallucination)
        # Ví dụ: GT 5 dòng, Pred 3 dòng -> Phạt 2 * 2.0 = -4.0
        row_diff = abs(len(gt_rows) - len(pred_rows))
        score -= (row_diff * 2.0) 

        # Để chấm điểm chính xác, ta cần mapping 1-1 giữa Pred và GT
        # Chiến thuật: Duyệt Pred để xem nó đúng hay sai (vì task extract cần check output)
        
        # Copy GT rows để đánh dấu những dòng đã được match
        available_gt_rows = gt_rows.copy()
        
        for p_row in pred_rows:
            if not isinstance(p_row, dict): continue
            
            p_fg = normalize_text(p_row.get("Mã FG"))
            p_qty = normalize_number_string(p_row.get("Số lượng"))
            
            # Tìm dòng match tốt nhất trong GT còn lại
            matched_idx = -1
            
            # Ưu tiên 1: Match đúng Mã FG
            candidates = []
            for idx, g_row in enumerate(available_gt_rows):
                if normalize_text(g_row.get("Mã FG")) == p_fg:
                    candidates.append(idx)
            
            # Nếu có nhiều dòng cùng mã FG (ít gặp nhưng có thể), check tiếp số lượng
            if len(candidates) == 1:
                matched_idx = candidates[0]
            elif len(candidates) > 1:
                # Chọn dòng có số lượng khớp
                best_sub_match = -1
                for c_idx in candidates:
                    g_qty_check = normalize_number_string(available_gt_rows[c_idx].get("Số lượng"))
                    if g_qty_check == p_qty:
                        best_sub_match = c_idx
                        break
                matched_idx = best_sub_match if best_sub_match != -1 else candidates[0]

            # --- CHẤM ĐIỂM DÒNG ---
            if matched_idx != -1:
                # TÌM THẤY MÃ FG TRONG GT -> Check tiếp Số lượng
                g_target = available_gt_rows[matched_idx]
                
                # 1. Mã FG Đúng (Base reward)
                score += 2.0 
                
                # 2. Check Số Lượng (CRITICAL)
                g_qty_target = normalize_number_string(g_target.get("Số lượng"))
                
                if p_qty == g_qty_target:
                    score += 3.0  # Thưởng lớn: Đúng cả Mã và Số lượng
                else:
                    score -= 5.0  # PHẠT RẤT NẶNG: Đúng mã nhưng sai số lượng (Nguy hiểm)
                
                # 3. Check ĐVT (Optional - Less important)
                # Chỉ thưởng, không phạt
                p_unit = normalize_text(p_row.get("Đvt"))
                g_unit = normalize_text(g_target.get("Đvt"))
                if smart_compare("Đvt", p_unit, g_unit) > 0.8:
                    score += 0.5
                
                # Xóa khỏi available để không match lại
                available_gt_rows.pop(matched_idx)
                
            else:
                # KHÔNG TÌM THẤY MÃ FG (Hallucination)
                # Model bịa ra một dòng mã hàng không có trong chứng từ
                score -= 5.0 

        rewards.append(score)
        
    return rewards