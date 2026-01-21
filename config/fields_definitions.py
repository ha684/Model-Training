# config/fields_definitions.py
# -*- coding: utf-8 -*-
"""
Central field definitions for all languages.
Add/remove fields here and prompts will automatically adjust.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

FieldType = Literal["metadata", "table"]
Language = Literal["en", "ja", "vi"]


@dataclass
class FieldDefinition:
    """Definition of a field with multilingual support."""
    key: str

    field_type: FieldType

    name_en: str
    name_ja: str
    name_vi: str

    description_en: str
    description_ja: str
    description_vi: str

    data_type: str = "string"

    optional: bool = False

    def get_name(self, language: Language) -> str:
        """Get field name for specified language."""
        return getattr(self, f"name_{language}")

    def get_description(self, language: Language) -> str:
        """Get field description for specified language."""
        return getattr(self, f"description_{language}")

FIELD_DEFINITIONS: List[FieldDefinition] = [
    FieldDefinition(
        key="document_number",
        field_type="metadata",
        name_en="document_number",
        name_ja="伝票番号",
        name_vi="Số phiếu",
        description_en="""- The main document/PO number, usually in top-right header area
   - Common formats: "PXB25110076", "2544051804257", "252.104.868.0263"
   - Extract exactly as shown with dots, dashes, spaces
   - NOT the handwritten page numbers like "1/2" or numbers after customer code""",
        description_ja="""- メインの伝票/PO番号、通常は右上のヘッダーエリアにあります
   - 一般的な形式: "PXB25110076", "2544051804257", "252.104.868.0263"
   - ドット、ダッシュ、スペースを含めて表示通りに正確に抽出
   - 手書きのページ番号("1/2"など)や顧客コードの後の番号ではありません""",
        description_vi="""- Số chứng từ/PO chính, thường ở góc trên bên phải
   - Các định dạng phổ biến: "PXB25110076", "2544051804257", "252.104.868.0263"
   - Trích xuất chính xác như hiển thị, giữ nguyên dấu chấm, gạch ngang, khoảng trắng
   - KHÔNG phải số trang viết tay như "1/2" hoặc số sau mã khách hàng""",
        data_type="string",
    ),
    FieldDefinition(
        key="document_date",
        field_type="metadata",
        name_en="document_date",
        name_ja="伝票日付",
        name_vi="Ngày chứng từ(dd/mm/yyyy)",
        description_en="""- Document issue date in dd/mm/yyyy format
   - Extract exactly as shown on document
   - If multiple dates exist, choose the one labeled as issue/document date""",
        description_ja="""- 伝票発行日をdd/mm/yyyy形式で
   - 伝票に表示されている通りに正確に抽出
   - 複数の日付がある場合は、発行日/伝票日とラベル付けされているものを選択""",
        description_vi="""- Ngày lập phiếu theo định dạng dd/mm/yyyy
   - Trích xuất chính xác như hiển thị trên chứng từ
   - Nếu có nhiều ngày, chọn ngày được ghi là ngày lập/ngày chứng từ""",
        data_type="date",
    ),
    FieldDefinition(
        key="export_date",
        field_type="metadata",
        name_en="export_date",
        name_ja="出庫日付",
        name_vi="Ngày xuất(dd/mm/yyyy)",
        description_en="""- Delivery or shipment date in dd/mm/yyyy format
   - Extract exactly as shown on document
   - May be same as or different from document_date""",
        description_ja="""- 配送または出荷日をdd/mm/yyyy形式で
   - 伝票に表示されている通りに正確に抽出
   - 伝票日付と同じ場合も異なる場合もあります""",
        description_vi="""- Ngày xuất kho hoặc ngày giao hàng theo định dạng dd/mm/yyyy
   - Trích xuất chính xác như hiển thị trên chứng từ
   - Có thể trùng hoặc khác với Ngày chứng từ""",
        data_type="date",
    ),
    FieldDefinition(
        key="receiving_warehouse",
        field_type="metadata",
        name_en="receiving_warehouse",
        name_ja="入庫倉庫",
        name_vi="Kho nhận",
        description_en="""- Name of warehouse or customer receiving the goods
   - Examples: "DC KHO HẢI PHÒNG", "Kho Hà Nội"
   - Extract full name as shown""",
        description_ja="""- 商品を受け取る倉庫または顧客の名前
   - 例: "DC KHO HẢI PHÒNG", "Kho Hà Nội"
   - 表示されている通りにフルネームを抽出""",
        description_vi="""- Tên kho hoặc khách hàng nhận hàng
   - Ví dụ: "DC KHO HẢI PHÒNG", "Kho Hà Nội"
   - Trích xuất tên đầy đủ như hiển thị""",
        data_type="string",
    ),
    FieldDefinition(
        key="address",
        field_type="metadata",
        name_en="address",
        name_ja="住所",
        name_vi="Địa chỉ",
        description_en="""- Full delivery address
   - Optional field - set null if not present on document
   - Do not guess or infer""",
        description_ja="""- 完全な配送先住所
   - オプションフィールド - 伝票に記載がない場合はnullに設定
   - 推測や推論はしない""",
        description_vi="""- Địa chỉ giao hàng đầy đủ
   - Trường tùy chọn - đặt null nếu không có trên chứng từ
   - Không đoán hoặc suy luận""",
        data_type="string",
        optional=True,
    ),
    FieldDefinition(
        key="ar_code",
        field_type="metadata",
        name_en="ar_code",
        name_ja="ARコード",
        name_vi="Mã AR",
        description_en="""- Customer account code
   - Examples: "A68/750", "A68/750_141", "AR1830"
   - Extract complete code including any suffix after underscore""",
        description_ja="""- 顧客アカウントコード
   - 例: "A68/750", "A68/750_141", "AR1830"
   - アンダースコアの後の接尾辞を含む完全なコードを抽出""",
        description_vi="""- Mã tài khoản khách hàng
   - Ví dụ: "A68/750", "A68/750_141", "AR1830"
   - Trích xuất mã đầy đủ bao gồm cả hậu tố sau dấu gạch dưới nếu có""",
        data_type="string",
    ),
    FieldDefinition(
        key="product_code",
        field_type="table",
        name_en="product_code",
        name_ja="製品コード",
        name_vi="Mã FG",
        description_en="""- Product identifier, typically leftmost column in table
   - Examples: "PG1490", "FC9345", "PL2368"
   - Extract exact alphanumeric code""",
        description_ja="""- 製品識別子、通常はテーブルの最も左の列
   - 例: "PG1490", "FC9345", "PL2368"
   - 正確な英数字コードを抽出""",
        description_vi="""- Mã sản phẩm, thường là cột ngoài cùng bên trái trong bảng
   - Ví dụ: "PG1490", "FC9345", "PL2368"
   - Trích xuất chính xác mã chữ và số""",
        data_type="string",
    ),
    FieldDefinition(
        key="unit",
        field_type="table",
        name_en="unit",
        name_ja="単位",
        name_vi="Đvt",
        description_en="""- Unit of measurement for the product
   - Examples: "Hộp" (box), "Gói" (pack), "Chai" (bottle), "Thùng" (carton)
   - Keep text as-is""",
        description_ja="""- 製品の測定単位
   - 例: "Hộp"(箱), "Gói"(パック), "Chai"(ボトル), "Thùng"(カートン)
   - テキストをそのまま保持""",
        description_vi="""- Đơn vị tính của sản phẩm
   - Ví dụ: "Hộp", "Gói", "Chai", "Thùng"
   - Giữ nguyên văn bản""",
        data_type="string",
    ),
    FieldDefinition(
        key="quantity",
        field_type="table",
        name_en="quantity",
        name_ja="数量",
        name_vi="Số lượng",
        description_en="""- Numeric quantity value
   - CRITICAL: Must be number type, not string
   - Example: {{"quantity": 168}} NOT {{"quantity": "168"}}""",
        description_ja="""- 数値の数量値
   - 重要: 文字列型ではなく数値型である必要があります
   - 例: {{"数量": 168}} であって {{"数量": "168"}} ではありません""",
        description_vi="""- Giá trị số lượng dạng số
   - QUAN TRỌNG: Phải là kiểu số (number), không phải chuỗi (string)
   - Ví dụ: {{"Số lượng": 168}} KHÔNG PHẢI {{"Số lượng": "168"}}""",
        data_type="number",
    ),
]


def get_fields_by_type(field_type: FieldType) -> List[FieldDefinition]:
    """Get all fields of a specific type."""
    return [f for f in FIELD_DEFINITIONS if f.field_type == field_type]


def get_field_names(
    language: Language, field_type: Optional[FieldType] = None
) -> List[str]:
    """Get field names for a specific language and optionally filter by type."""
    fields = FIELD_DEFINITIONS if field_type is None else get_fields_by_type(field_type)
    return [f.get_name(language) for f in fields]
