"""
Reward functions for Qwen3 VL - Split metadata and table_rows scoring.
"""
import json
import re
from difflib import SequenceMatcher

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

def normalize_date(val):
    if val is None: return ""
    s = str(val).strip()
    s = s.replace("-", "/")
    s = s.replace(" ", "")
    return s

def normalize_ar_code(val):
    if val is None: return ""
    s = str(val).strip()
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

    if "ngày" in key_lower or "date" in key_lower:
        p_date = normalize_date(p)
        g_date = normalize_date(g)
        return 1.0 if p_date == g_date else 0.0

    if "số phiếu" in key_lower or "invoice" in key_lower:
        p_clean = normalize_number_string(p)
        g_clean = normalize_number_string(g)
        return 1.0 if p_clean == g_clean else 0.0

    if "mã ar" in key_lower or "mã kh" in key_lower:
        p_clean = normalize_ar_code(p)
        g_clean = normalize_ar_code(g)
        return 1.0 if p_clean == g_clean else 0.0

    if "địa chỉ" in key_lower or "kho" in key_lower or "tên" in key_lower or "address" in key_lower:
        if p == g: return 1.0
        sim = similarity_ratio(p.lower(), g.lower())
        return sim if sim > 0.85 else 0.0

    return 1.0 if p == g else 0.0

def is_float_match(val1, val2):
    if val1 is None or val2 is None: return False
    try:
        s1 = str(val1).replace(",", "").strip()
        s2 = str(val2).replace(",", "").strip()
        
        f1 = float(s1)
        f2 = float(s2)
        
        return abs(f1 - f2) < 0.01
    except:
        return False

def format_reward_func(completions, **kwargs):
    """Reward for valid JSON format."""
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

def metadata_accuracy_reward_func(completions, answer, **kwargs):
    """Reward specifically for metadata extraction accuracy."""
    rewards = []
    
    for completion, gt_item in zip(completions, answer):
        score = 0.0

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

        gt_meta = gt_data.get("metadata", {})
        pred_meta = pred_data.get("metadata", {})
        
        if isinstance(gt_meta, list): gt_meta = gt_meta[0] if gt_meta else {}
        if isinstance(pred_meta, list): pred_meta = pred_meta[0] if pred_meta else {}

        mapped_pred_keys = set()
        total_gt_fields = 0

        for k_gt, v_gt in gt_meta.items():
            if normalize_text(v_gt) is None: continue
            total_gt_fields += 1

            best_k_pred = None
            
            if k_gt in pred_meta:
                best_k_pred = k_gt
            
            if best_k_pred:
                mapped_pred_keys.add(best_k_pred)
                v_pred = pred_meta[best_k_pred]
                
                match_val = check_field_match(k_gt, v_pred, v_gt)

                if match_val == 1.0: 
                    score += 10.0
                elif match_val > 0.85: 
                    score += 5.0
                else: 
                    score -= 7.0
            else:
                score -= 7.0

        for k_p in pred_meta.keys():
            if k_p not in mapped_pred_keys:
                if normalize_text(pred_meta[k_p]) is not None:
                    score -= 7.0

        rewards.append(score)
        
    return rewards

def table_rows_accuracy_reward_func(completions, answer, **kwargs):
    """Reward specifically for table rows extraction accuracy."""
    rewards = []
    
    for completion, gt_item in zip(completions, answer):
        score = 0.0

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

        gt_rows = gt_data.get("table_rows", [])
        pred_rows = pred_data.get("table_rows", [])
        
        if not isinstance(gt_rows, list): gt_rows = []
        if not isinstance(pred_rows, list): pred_rows = []

        if len(gt_rows) == 0:
            # No ground truth rows - neutral score if no predictions
            if len(pred_rows) == 0:
                rewards.append(0.0)
            else:
                # Penalty for hallucinated rows
                rewards.append(-7.0 * len(pred_rows))
            continue

        matched_gt_indices = set()
        
        # Build map for efficient lookup
        gt_fg_map = {}
        for i, r in enumerate(gt_rows):
            fg = normalize_text(r.get("Mã FG"))
            if fg:
                if fg not in gt_fg_map:
                    gt_fg_map[fg] = []
                gt_fg_map[fg].append(i)
        
        # Match predicted rows to ground truth
        for p_row in pred_rows:
            if not isinstance(p_row, dict): 
                continue
            
            p_fg = normalize_text(p_row.get("Mã FG"))
            p_qty_raw = p_row.get("Số lượng")
            
            if p_fg and p_fg in gt_fg_map:
                candidates = [i for i in gt_fg_map[p_fg] if i not in matched_gt_indices]
                
                if candidates:
                    best_match_idx = candidates[0]
                    
                    # If multiple candidates, prefer quantity match
                    if len(candidates) > 1:
                        for idx in candidates:
                            g_qty = gt_rows[idx].get("Số lượng")
                            if is_float_match(p_qty_raw, g_qty):
                                best_match_idx = idx
                                break
                    
                    matched_gt_indices.add(best_match_idx)
                    g_qty = gt_rows[best_match_idx].get("Số lượng")
                    
                    # Reward for correct FG code
                    score += 7.0
                    
                    # Bonus/penalty for quantity accuracy
                    if is_float_match(p_qty_raw, g_qty):
                        score += 3.0
                    else:
                        score -= 5.0
                else:
                    # FG code exists but already matched (duplicate)
                    score -= 7.0
            else:
                # Hallucinated or wrong FG code
                score -= 7.0

        # Penalty for missed rows
        missed_count = len(gt_rows) - len(matched_gt_indices)
        if missed_count > 0:
            score -= (missed_count * 7.0)

        rewards.append(score)
        
    return rewards

def length_penalty_reward_func(completions, answer, **kwargs):
    """Penalty for outputs that are too long or too short compared to ground truth."""
    rewards = []
    
    for completion, gt_item in zip(completions, answer):
        gt_str = gt_item if isinstance(gt_item, str) else json.dumps(gt_item)
        gt_len = len(gt_str.replace(" ", "").replace("\n", ""))

        pred_len = len(completion.replace(" ", "").replace("\n", ""))

        if gt_len == 0: gt_len = 1
        
        ratio = pred_len / gt_len
        score = 0.0

        if ratio > 2.0:
            score = -4.0 
        elif ratio > 1.5:
            score = -2.0
        elif ratio < 0.5:
            score = -4.0
        elif ratio < 0.7:
            score = -2.0
        else:
            score = 0.5
            
        rewards.append(score)
        
    return rewards