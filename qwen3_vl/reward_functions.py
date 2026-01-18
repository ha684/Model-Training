import json
import re
from difflib import SequenceMatcher

# ==============================================================================
# 1. HELPER FUNCTIONS (UPDATED LOGIC)
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
    """Giữ lại chỉ số và chữ cái (cho Số phiếu, Số điện thoại)"""
    if val is None: return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(val))

def normalize_date(val):
    """
    Chuẩn hóa ngày tháng:
    - Thay thế dấu gạch ngang (-) bằng gạch chéo (/)
    - Xóa khoảng trắng thừa
    Ví dụ: '20-10 - 2023' -> '20/10/2023'
    """
    if val is None: return ""
    s = str(val).strip()
    # Thay thế - bằng /
    s = s.replace("-", "/")
    # Xóa khoảng trắng nếu OCR bị tách rời (vd: 20 / 10)
    s = s.replace(" ", "")
    return s

def normalize_ar_code(val):
    """
    Xử lý đặc thù Mã AR linh hoạt:
    - Cắt bỏ phần đuôi sau dấu gạch dưới (_) hoặc khoảng trắng ( )
    - Chấp nhận 1 hay nhiều dấu _, dấu cách đều được xử lý
    Ví dụ: 
      'AR1830__141' -> 'AR1830'
      'AR1830_141'  -> 'AR1830'
      'AR1830 141'  -> 'AR1830'
    """
    if val is None: return ""
    s = str(val).strip()
    # Regex split: Tách tại bất kỳ chuỗi nào chứa _ hoặc khoảng trắng
    parts = re.split(r'[_\s]+', s)
    return parts[0].strip() if parts else s

def similarity_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def check_field_match(key, pred_val, gt_val):
    p = normalize_text(pred_val)
    g = normalize_text(gt_val)
    
    if p is None and g is None: return 1.0
    if p is None or g is None: return 0.0

    key_lower = key.lower()

    # --- NHÓM 1: DATE (Chấp nhận / và -) ---
    if "ngày" in key_lower or "date" in key_lower:
        p_date = normalize_date(p)
        g_date = normalize_date(g)
        return 1.0 if p_date == g_date else 0.0

    # --- NHÓM 2: NORMALIZED ALPHANUM (Số phiếu) ---
    if "số phiếu" in key_lower or "invoice" in key_lower:
        p_clean = normalize_number_string(p)
        g_clean = normalize_number_string(g)
        return 1.0 if p_clean == g_clean else 0.0

    # --- NHÓM 3: PREFIX MATCH (Mã AR, Mã KH) ---
    # Logic mới: Cắt bỏ phần mở rộng sau _ hoặc space
    if "mã ar" in key_lower or "mã kh" in key_lower:
        p_clean = normalize_ar_code(p)
        g_clean = normalize_ar_code(g)
        return 1.0 if p_clean == g_clean else 0.0

    # --- NHÓM 4: FUZZY MATCH (Kho, Địa chỉ) ---
    if "địa chỉ" in key_lower or "kho" in key_lower or "tên" in key_lower or "address" in key_lower:
        if p == g: return 1.0
        sim = similarity_ratio(p.lower(), g.lower())
        return sim if sim > 0.85 else 0.0

    # --- DEFAULT: EXACT MATCH ---
    return 1.0 if p == g else 0.0

# ==============================================================================
# 2. REWARD FUNCTIONS (KEEPING "NO ESCAPE" STRATEGY)
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
    
    for completion, gt_item in zip(completions, answer):
        score = 0.0
        
        # --- Parse Data ---
        gt_data = gt_item
        if isinstance(gt_item, str):
            try: gt_data = json.loads(gt_item)
            except: rewards.append(-5.0); continue 
            
        if isinstance(gt_data, list) and len(gt_data) > 0: gt_data = gt_data[0]
        if not isinstance(gt_data, dict): gt_data = {}

        pred_data = extract_json_from_text(completion)
        if pred_data is None: rewards.append(-5.0); continue
            
        if isinstance(pred_data, list) and len(pred_data) > 0: pred_data = pred_data[0]
        if not isinstance(pred_data, dict): rewards.append(-5.0); continue

        # ======================================================================
        # 1. METADATA SCORING
        # ======================================================================
        gt_meta = gt_data.get("metadata", {})
        pred_meta = pred_data.get("metadata", {})
        
        if isinstance(gt_meta, list): gt_meta = gt_meta[0] if gt_meta else {}
        if isinstance(pred_meta, list): pred_meta = pred_meta[0] if pred_meta else {}

        # Check fields exist in GT
        for k, g_v in gt_meta.items():
            if normalize_text(g_v) is None: continue
            
            p_v = pred_meta.get(k)
            match_val = check_field_match(k, p_v, g_v)
            
            if match_val == 1.0:
                score += 2.0 
            elif match_val > 0.85:
                score += 1.0 
            else:
                score -= 2.0 # Phạt sai Metadata

        # Check Hallucination Metadata
        for k, p_v in pred_meta.items():
            if k not in gt_meta and normalize_text(p_v) is not None:
                score -= 1.0

        # ======================================================================
        # 2. TABLE ROWS SCORING
        # ======================================================================
        gt_rows = gt_data.get("table_rows", [])
        pred_rows = pred_data.get("table_rows", [])
        
        if not isinstance(gt_rows, list): gt_rows = []
        if not isinstance(pred_rows, list): pred_rows = []

        matched_gt_indices = set()
        
        # Check Pred vs GT
        for p_row in pred_rows:
            if not isinstance(p_row, dict): continue
            
            p_fg = normalize_text(p_row.get("Mã FG"))
            p_qty = normalize_number_string(p_row.get("Số lượng"))
            
            # Match Logic: Tìm dòng GT có cùng Mã FG
            best_match_idx = -1
            found_match = False
            
            candidates = [i for i, r in enumerate(gt_rows) 
                          if normalize_text(r.get("Mã FG")) == p_fg 
                          and i not in matched_gt_indices]

            if candidates:
                best_match_idx = candidates[0] 
                # Nếu nhiều candidate, tìm cái khớp số lượng
                for idx in candidates:
                    if normalize_number_string(gt_rows[idx].get("Số lượng")) == p_qty:
                        best_match_idx = idx
                        break
                found_match = True
                matched_gt_indices.add(best_match_idx)

            if found_match:
                # Check Số Lượng
                g_target = gt_rows[best_match_idx]
                g_qty_target = normalize_number_string(g_target.get("Số lượng"))

                if p_qty == g_qty_target:
                    score += 5.0 
                else:
                    score -= 5.0 # Phạt Đúng Mã - Sai Số Lượng
            else:
                score -= 5.0 # Phạt Hallucination Table Row

        # Check Missing GT Rows
        missed_count = len(gt_rows) - len(matched_gt_indices)
        if missed_count > 0:
            score -= (missed_count * 5.0) # Phạt Missing

        rewards.append(score)
        
    return rewards