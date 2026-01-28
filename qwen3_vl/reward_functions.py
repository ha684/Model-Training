import json
import re

from rapidfuzz import fuzz

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
    try:
        return json.loads(text)
    except:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except:
            pass
    return None


def normalize_key(key):
    if not isinstance(key, str):
        return str(key)
    return CANONICAL_KEY_MAP.get(key.lower().strip(), key)


def normalize_text(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ["null", "none", "nan", ""]:
        return None
    return s


def normalize_number_str(val):
    if val is None:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(val))


def is_float_equal(val1, val2):
    if val1 is None or val2 is None:
        return False
    try:
        s1 = str(val1).replace(",", "").strip()
        s2 = str(val2).replace(",", "").strip()
        return abs(float(s1) - float(s2)) < 0.01
    except:
        return False


def normalize_date(val):
    if val is None:
        return ""
    s = str(val).strip()
    s = s.replace("-", "/")
    s = s.replace(" ", "")
    return s


def normalize_ar_code(val):
    if val is None:
        return ""
    s = str(val).strip()
    parts = re.split(r"[_\s]+", s)
    return parts[0].strip() if parts else s


def check_metadata_field(key, pred_val, gt_val):
    p = normalize_text(pred_val)
    g = normalize_text(gt_val)

    if p is None and g is None:
        return 1.0
    if p is None or g is None:
        return 0.0

    key_lower = key.lower()

    if "date" in key_lower:
        return 1.0 if normalize_date(p) == normalize_date(g) else 0.0

    if key == "document_number":
        return 1.0 if normalize_number_str(p) == normalize_number_str(g) else 0.0

    if key == "ar_code":
        return 1.0 if normalize_ar_code(p) == normalize_ar_code(g) else 0.0

    if key in ["receiving_warehouse", "address"]:
        if p == g:
            return 1.0
        score = fuzz.ratio(p.lower(), g.lower())
        return 1.0 if score > 85 else 0.0

    return 1.0 if p == g else 0.0


def normalize_json_structure(data):
    if not isinstance(data, dict):
        return {"metadata": {}, "table_rows": []}

    raw_meta = data.get("metadata", {})
    if isinstance(raw_meta, list):
        raw_meta = raw_meta[0] if raw_meta else {}
    norm_meta = (
        {normalize_key(k): v for k, v in raw_meta.items()}
        if isinstance(raw_meta, dict)
        else {}
    )

    raw_rows = data.get("table_rows", [])
    if not isinstance(raw_rows, list):
        raw_rows = []
    norm_rows = []
    for r in raw_rows:
        if isinstance(r, dict):
            norm_rows.append({normalize_key(k): v for k, v in r.items()})

    return {"metadata": norm_meta, "table_rows": norm_rows}


def format_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        data = extract_json_from_text(completion)
        if data is None:
            rewards.append(-2.0)
        else:
            has_metadata = isinstance(data.get("metadata"), (dict, list))
            has_table = isinstance(data.get("table_rows"), list)
            score = 0.0
            if has_metadata and has_table:
                score = 0.5
            rewards.append(score)
    return rewards


def metadata_reward_func(completions, answer, **kwargs):
    rewards = []

    METADATA_SCALE = 6.0

    VALID_SCHEMA_KEYS = set(ALL_META_KEYS)

    for completion, gt_item in zip(completions, answer):
        pred_raw = extract_json_from_text(completion)
        if pred_raw is None:
            rewards.append(-1.0 * METADATA_SCALE * 1.5)
            continue

        gt_raw = gt_item if isinstance(gt_item, dict) else json.loads(gt_item)
        pred_data = normalize_json_structure(pred_raw)
        gt_data = normalize_json_structure(gt_raw)

        gt_meta = gt_data["metadata"]
        pred_meta = pred_data["metadata"]

        gt_active_keys = [
            k for k in ALL_META_KEYS if normalize_text(gt_meta.get(k)) is not None
        ]

        extra_valid_keys = []
        for k in ALL_META_KEYS:
            gt_val = normalize_text(gt_meta.get(k))
            pred_val = normalize_text(pred_meta.get(k))
            if gt_val is None and pred_val is not None:
                extra_valid_keys.append(k)

        ghost_keys = [k for k in pred_meta.keys() if k not in VALID_SCHEMA_KEYS]

        match_score_sum = 0.0
        for k in gt_active_keys:
            gt_val = gt_meta.get(k)
            pred_val = pred_meta.get(k)
            match_score_sum += check_metadata_field(k, pred_val, gt_val)

        denominator = len(gt_active_keys) + len(extra_valid_keys) + len(ghost_keys)

        if denominator == 0:
            final_score = 1.0
        else:
            final_score = match_score_sum / denominator

        rewards.append(((final_score * 2.0) - 1.0) * METADATA_SCALE)

    return rewards


def table_row_reward_func(completions, answer, **kwargs):
    rewards = []

    BASE_SCALE = 5.0

    ROW_MATCH_BONUS = 0.5

    ROW_WRONG_PENALTY = 1.0

    ROW_MISSING_PENALTY = 1.0

    for completion, gt_item in zip(completions, answer):
        pred_raw = extract_json_from_text(completion)
        if pred_raw is None:
            rewards.append(-5.0)
            continue

        gt_raw = gt_item if isinstance(gt_item, dict) else json.loads(gt_item)
        pred_data = normalize_json_structure(pred_raw)
        gt_data = normalize_json_structure(gt_raw)

        gt_rows = gt_data["table_rows"]
        pred_rows = pred_data["table_rows"]

        gt_map = {}
        for r in gt_rows:
            pc = normalize_text(r.get("product_code"))
            if not pc:
                continue
            gt_map[pc] = r.get("quantity")

        total_gt = len(gt_map)

        if total_gt == 0:
            if len(pred_rows) == 0:
                rewards.append(BASE_SCALE)
            else:
                rewards.append(-1.0 * len(pred_rows) * ROW_WRONG_PENALTY)
            continue

        matched_count = 0
        wrong_count = 0
        pred_codes_seen = set()

        for r in pred_rows:
            p_code = normalize_text(r.get("product_code"))
            p_qty = r.get("quantity")

            if not p_code or p_code in pred_codes_seen:
                wrong_count += 1
                continue
            pred_codes_seen.add(p_code)

            if p_code in gt_map:
                g_qty = gt_map[p_code]
                if g_qty is not None and is_float_equal(p_qty, g_qty):
                    matched_count += 1
                else:
                    wrong_count += 1
            else:
                wrong_count += 1

        missing_count = total_gt - matched_count

        precision = matched_count / len(pred_rows) if len(pred_rows) > 0 else 0
        recall = matched_count / total_gt if total_gt > 0 else 0
        f1 = 0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        bonus_score = matched_count * ROW_MATCH_BONUS

        penalty_score = (wrong_count * ROW_WRONG_PENALTY) + (
            missing_count * ROW_MISSING_PENALTY
        )

        final_score = (f1 * BASE_SCALE) + bonus_score - penalty_score

        final_score = max(final_score, -10.0)

        rewards.append(final_score)

    return rewards


def length_penalty_reward_func(completions, answer, **kwargs):
    """
    Penalty if the output length is significantly different from the Ground Truth.
    Helps prevent infinite loops or too short outputs.
    """
    rewards = []

    for completion, gt_item in zip(completions, answer):
        gt_str = gt_item if isinstance(gt_item, str) else json.dumps(gt_item)
        gt_len = len(gt_str.replace(" ", "").replace("\n", ""))

        pred_len = len(completion.replace(" ", "").replace("\n", ""))

        if gt_len == 0:
            gt_len = 1

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
