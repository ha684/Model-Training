import json
import re
from difflib import SequenceMatcher

FIELD_MAPPING_VN_TO_EN = {
    "Số phiếu": "document_number",
    "Ngày chứng từ(dd/mm/yyyy)": "document_date",
    "Ngày xuất(dd/mm/yyyy)": "export_date",
    "Kho nhận": "receiving_warehouse",
    "Địa chỉ": "address",
    "Mã AR": "ar_code",
    "Mã FG": "product_code",
    "Đvt": "unit",
    "Số lượng": "quantity",
}

FIELD_MAPPING_VN_TO_JA = {
    "Số phiếu": "伝票番号",
    "Ngày chứng từ(dd/mm/yyyy)": "伝票日付",
    "Ngày xuất(dd/mm/yyyy)": "出庫日付",
    "Kho nhận": "入庫倉庫",
    "Địa chỉ": "住所",
    "Mã AR": "ARコード",
    "Mã FG": "製品コード",
    "Đvt": "単位",
    "Số lượng": "数量",
}

TABLE_KEYS = {"product_code", "unit", "quantity"}
ALL_META_KEYS = [v for v in FIELD_MAPPING_VN_TO_EN.values() if v not in TABLE_KEYS]

CANONICAL_KEY_MAP = {}
for k_vn, k_en in FIELD_MAPPING_VN_TO_EN.items():
    CANONICAL_KEY_MAP[k_vn.lower()] = k_en
    CANONICAL_KEY_MAP[k_en.lower()] = k_en
for k_vn, k_ja in FIELD_MAPPING_VN_TO_JA.items():
    if k_vn in FIELD_MAPPING_VN_TO_EN:
        CANONICAL_KEY_MAP[k_ja.lower()] = FIELD_MAPPING_VN_TO_EN[k_vn]

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

def normalize_key(key):
    if not isinstance(key, str): return str(key)
    return CANONICAL_KEY_MAP.get(key.lower().strip(), key)

def normalize_text(val):
    if val is None: return None
    s = str(val).strip()
    if s.lower() in ["null", "none", "nan", ""]: return None
    return s

def normalize_number_str(val):
    if val is None: return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(val))

def is_float_equal(val1, val2):
    if val1 is None or val2 is None: return False
    try:
        s1 = str(val1).replace(",", "").strip()
        s2 = str(val2).replace(",", "").strip()
        return abs(float(s1) - float(s2)) < 0.01
    except:
        return False

def check_metadata_field(key, pred_val, gt_val):
    p = normalize_text(pred_val)
    g = normalize_text(gt_val)
    
    if p is None and g is None: return 1.0
    if p is None or g is None: return 0.0
    
    if key in ["document_number", "ar_code"]:
        return 1.0 if normalize_number_str(p) == normalize_number_str(g) else 0.0
    
    if key in ["receiving_warehouse", "address"]:
        if p == g: return 1.0
        return 1.0 if SequenceMatcher(None, p.lower(), g.lower()).ratio() > 0.85 else 0.0
        
    return 1.0 if p == g else 0.0

def normalize_json_structure(data):
    if not isinstance(data, dict): return {"metadata": {}, "table_rows": []}
    
    raw_meta = data.get("metadata", {})
    if isinstance(raw_meta, list): raw_meta = raw_meta[0] if raw_meta else {}
    norm_meta = {normalize_key(k): v for k, v in raw_meta.items()} if isinstance(raw_meta, dict) else {}
    
    raw_rows = data.get("table_rows", [])
    if not isinstance(raw_rows, list): raw_rows = []
    norm_rows = []
    for r in raw_rows:
        if isinstance(r, dict):
            norm_rows.append({normalize_key(k): v for k, v in r.items()})
            
    return {"metadata": norm_meta, "table_rows": norm_rows}

def metadata_reward_func(completions, answer, **kwargs):
    rewards = []
    
    for completion, gt_item in zip(completions, answer):
        pred_raw = extract_json_from_text(completion)
        if pred_raw is None:
            rewards.append(-1.0)
            continue
            
        gt_raw = gt_item if isinstance(gt_item, dict) else json.loads(gt_item)
            
        pred_data = normalize_json_structure(pred_raw)
        gt_data = normalize_json_structure(gt_raw)
        
        gt_meta = gt_data["metadata"]
        pred_meta = pred_data["metadata"]
        
        meta_matches = 0
        total_meta_keys = len(ALL_META_KEYS)
        
        for k in ALL_META_KEYS:
            gt_val = gt_meta.get(k)
            pred_val = pred_meta.get(k)
            meta_matches += check_metadata_field(k, pred_val, gt_val)
        
        meta_score = meta_matches / total_meta_keys
        scaled_reward = (meta_score * 2.0) - 1.0
        
        rewards.append(scaled_reward)
        
    return rewards

def table_row_reward_func(completions, answer, **kwargs):
    rewards = []
    
    for completion, gt_item in zip(completions, answer):
        pred_raw = extract_json_from_text(completion)
        if pred_raw is None:
            rewards.append(-1.0)
            continue
            
        gt_raw = gt_item if isinstance(gt_item, dict) else json.loads(gt_item)
            
        pred_data = normalize_json_structure(pred_raw)
        gt_data = normalize_json_structure(gt_raw)
        
        gt_rows = gt_data["table_rows"]
        pred_rows = pred_data["table_rows"]
        
        gt_map = {}
        for r in gt_rows:
            pc = normalize_text(r.get("product_code"))
            qty = r.get("quantity")
            if pc: gt_map[pc] = qty
            
        matched_count = 0
        matched_gt_codes = set()
        
        for r in pred_rows:
            p_code = normalize_text(r.get("product_code"))
            p_qty = r.get("quantity")
            
            if p_code and p_code in gt_map:
                if p_code not in matched_gt_codes:
                    g_qty = gt_map[p_code]
                    if is_float_equal(p_qty, g_qty):
                        matched_count += 1
                        matched_gt_codes.add(p_code)
        
        n_gt = len(gt_map)
        n_pred = len(pred_rows)
        
        if n_gt + n_pred == 0:
            table_score = 1.0
        else:
            table_score = (2 * matched_count) / (n_gt + n_pred)
            
        scaled_reward = (table_score * 2.0) - 1.0
        
        rewards.append(scaled_reward)
        
    return rewards