"""
industrial_auditor.py — Bộ Giám Định Rác Thải Công Nghiệp & Mức Độ Nhiễm Bẩn
(Industrial Waste Material & Contamination Auditing Engine)

Tuân thủ:
  - Luật Bảo vệ Môi trường Việt Nam 2020 (Hiệu lực phân loại rác tại nguồn)
  - Tiêu chuẩn Mã nhận diện nhựa quốc tế (ASTM D7611 / SPI Resin Identification Code)
  - Tiêu chuẩn Phân loại rác của Tổ chức Tái chế Quốc tế (BIR - Bureau of International Recycling)
"""

from typing import Dict, Any, List

# ── CƠ SỞ TRI THỨC VẬT LIỆU CÔNG NGHIỆP & RỦI RO THỰC TẾ ──
INDUSTRIAL_KNOWLEDGE_BASE = {
    # ── 1. NHÓM NHỰA & BAO BÌ (PLASTICS & PACKAGING) ──
    "plastic bottle": {
        "vn_name": "Chai Nhựa Uống Nước",
        "resin_code": "RIC #1 (PET - Polyethylene Terephthalate)",
        "material": "Nhựa nhiệt dẻo PET nguyên sinh / tái sinh",
        "contamination_score": 10,
        "is_recyclable": True,
        "target_stream": "🟢 Tái Chế Tinh (Mechanical Recycling)",
        "hazard_risk": "Không nguy hại (An toàn)",
        "hazard_level": "None",
        "action": "Trút bỏ nước thừa, tháo nắp PP riêng, bẹp thân chai đưa vào dây chuyền ép kiện."
    },
    "plastic water bottle": {
        "vn_name": "Chai Nước Khoáng Trong Suốt",
        "resin_code": "RIC #1 (PET)",
        "material": "Nhựa PET có độ tinh khiết cao nhất",
        "contamination_score": 5,
        "is_recyclable": True,
        "target_stream": "🟢 Tái Chế Tinh (Hạt Nhựa Cấp Thực Phẩm rPET)",
        "hazard_risk": "Không nguy hại",
        "hazard_level": "None",
        "action": "Giá trị tái chế cao nhất. Rửa sạch, ép xẹp."
    },
    "boba cup": {
        "vn_name": "Cốc Trà Sữa / Ly Nhựa Dùng 1 Lần",
        "resin_code": "RIC #5 (PP) hoặc RIC #1 (PET)",
        "material": "Nhựa PP/PET kèm màng dán nắp composite",
        "contamination_score": 65,
        "is_recyclable": False,
        "target_stream": "⚪ Đốt Rác Phát Điện (RDF / Waste-to-Energy)",
        "hazard_risk": "Độ dính bẩn cao (Đường sữa gây nấm mốc)",
        "hazard_level": "Low",
        "action": "Còn dính cặn đường sữa. Nếu chưa rửa sạch: Loại khỏi luồng tái chế, đưa vào lò đốt phát điện."
    },
    "plastic cup": {
        "vn_name": "Cốc / Ly Nhựa",
        "resin_code": "RIC #5 (PP) / RIC #1 (PET)",
        "material": "Polypropylene mỏng",
        "contamination_score": 40,
        "is_recyclable": True,
        "target_stream": "🟢 Tái Chế Cơ Học (Nếu rửa sạch)",
        "hazard_risk": "Không nguy hại",
        "hazard_level": "None",
        "action": "Rửa sạch cặn đồ uống để đạt chuẩn tái chế."
    },
    "styrofoam food container": {
        "vn_name": "Hộp Xốp Đựng Cơm Dính Dầu Mỡ",
        "resin_code": "RIC #6 (PS - Expanded Polystyrene)",
        "material": "Xốp PS xốp tỷ trọng siêu nhẹ",
        "contamination_score": 85,
        "is_recyclable": False,
        "target_stream": "⚪ Đốt Rác Phát Điện (RDF) / Chôn Lấp Hợp Vệ Sinh",
        "hazard_risk": "Dầu mỡ thấm sâu vào cấu trúc bọt xốp không thể rửa sạch",
        "hazard_level": "Low",
        "action": "⚠️ TUYỆT ĐỐI KHÔNG ĐƯA VÀO LUỒNG TÁI CHẾ. Dầu mỡ sẽ làm hỏng cả tấn hạt nhựa. Chuyển lò thiêu đốt RDF."
    },
    "dirty foam box": {
        "vn_name": "Hộp Xốp Thức Ăn Bẩn",
        "resin_code": "RIC #6 (PS)",
        "material": "Polystyrene xốp ngấm chất béo",
        "contamination_score": 90,
        "is_recyclable": False,
        "target_stream": "⚪ Đốt Rác Phát Điện (RDF)",
        "hazard_risk": "Ô nhiễm hữu cơ nghiêm trọng",
        "hazard_level": "Low",
        "action": "Đưa vào lò đốt thu hồi năng lượng (Waste-to-Energy)."
    },
    "plastic grocery bag": {
        "vn_name": "Túi Nilon / Bao Bì Mềm",
        "resin_code": "RIC #4 (LDPE - Low Density Polyethylene)",
        "material": "Màng nhựa mềm dẻo",
        "contamination_score": 45,
        "is_recyclable": True,
        "target_stream": "🟢 Tái Chế Màng Film (Ép đùn nhiệt)",
        "hazard_risk": "Dễ quấn kẹt trục máy băm công nghiệp",
        "hazard_level": "Low",
        "action": "Gom riêng dạng cuộn. Không vứt lẫn với rác cứng."
    },
    "snack packaging bag": {
        "vn_name": "Bao Bì Snack / Vỏ Bánh Mì Tôm",
        "resin_code": "RIC #7 (OTHER - Multi-layer Laminate)",
        "material": "Màng nhôm ghép nhựa phức hợp (Metallized Film)",
        "contamination_score": 70,
        "is_recyclable": False,
        "target_stream": "⚪ Đốt Rác Phát Điện (RDF / Nhiên Liệu Thay Thế)",
        "hazard_risk": "Không thể phân tách nhôm và nhựa bằng phương pháp cơ học",
        "hazard_level": "None",
        "action": "Không thể tái chế thông thường. Chuyển sang nhà máy xi măng làm nhiên liệu đồng xử lý."
    },

    # ── 2. NHÓM KIM LOẠI (METALS) ──
    "aluminum can": {
        "vn_name": "Vỏ Lon Nhôm Nước Ngọt",
        "resin_code": "Kim Loại Màu (Aluminum Alloy 3004/3104)",
        "material": "Hợp kim nhôm tinh luyện",
        "contamination_score": 15,
        "is_recyclable": True,
        "target_stream": "🟢 Tái Chế Kim Loại Vô Hạn Lần (Closed-Loop Recycling)",
        "hazard_risk": "Không nguy hại",
        "hazard_level": "None",
        "action": "Nén dẹp vỏ lon. Tái chế nhôm tiết kiệm 95% năng lượng so với luyện quặng bauxit."
    },
    "crushed can": {
        "vn_name": "Vỏ Lon Bẹp Dúm",
        "resin_code": "Kim Loại (Alloy Can)",
        "material": "Hợp kim nhôm / Thép mạ thiếc",
        "contamination_score": 20,
        "is_recyclable": True,
        "target_stream": "🟢 Tái Chế Kim Loại Tinh",
        "hazard_risk": "Gờ kim loại sắc nhọn",
        "hazard_level": "Low",
        "action": "Giá trị kinh tế cao, đưa thẳng vào lò nung luyện kim."
    },

    # ── 3. NHÓM GIẤY & BAO BÌ TRÁNG PE (PAPER & CARTON) ──
    "cardboard box": {
        "vn_name": "Thùng Bìa Carton",
        "resin_code": "OCC (Old Corrugated Containers)",
        "material": "Sợi cellulose gỗ ép gợn sóng",
        "contamination_score": 10,
        "is_recyclable": True,
        "target_stream": "🟢 Tái Chế Giấy & Bao Bì Logistics",
        "hazard_risk": "Dễ hút ẩm làm giảm phẩm cấp sợi giấy",
        "hazard_level": "None",
        "action": "Tháo gỡ băng keo dính, gấp phẳng giữ khô ráo."
    },
    "paper cup": {
        "vn_name": "Ly Giấy Cà Phê / Trà Sữa",
        "resin_code": "Giấy tráng PE (Poly-coated Paperboard)",
        "material": "95% sợi giấy Kraft tráng 5% màng nhựa PE ngăn thấm",
        "contamination_score": 60,
        "is_recyclable": False,
        "target_stream": "⚪ Xử Lý Tách Lớp Chuyên Dụng / Lò Đốt RDF",
        "hazard_risk": "Màng nhựa dính chặt không tan trong bể nấu bột giấy",
        "hazard_level": "None",
        "action": "⚠️ Người dùng thường nhầm là giấy thông thường. Cần dây chuyền thủy lực tách màng riêng."
    },
    "greasy pizza box": {
        "vn_name": "Hộp Pizza Thấm Dầu Mỡ",
        "resin_code": "Carton Ô Nhiễm Dầu Mỡ",
        "material": "Bìa carton ngấm mỡ động thực vật",
        "contamination_score": 80,
        "is_recyclable": False,
        "target_stream": "🟠 Ủ Phân Vi Sinh Compost / Lò Đốt RDF",
        "hazard_risk": "Dầu mỡ phá vỡ liên kết hydrogen của sợi giấy khi nấu bột",
        "hazard_level": "None",
        "action": "Xé bỏ phần nắp sạch đem tái chế; phần đáy dính mỡ chuyển ủ phân compost hoặc đốt RDF."
    },
    "milk carton": {
        "vn_name": "Hộp Sữa Nhiều Lớp (Tetra Pak)",
        "resin_code": "Aseptic Composite (75% Giấy + 20% PE + 5% Nhôm)",
        "material": "Bao bì vô trùng 6 lớp bảo vệ",
        "contamination_score": 30,
        "is_recyclable": True,
        "target_stream": "🟢 Nhà Máy Tái Chế Tách Bột Giấy & Ép Tấm Lợp PolyAl",
        "hazard_risk": "Lên men bốc mùi nếu còn sữa thừa",
        "hazard_level": "None",
        "action": "Mở tai hộp, ép bẹp, đưa về điểm thu gom Tetra Pak chuyên biệt."
    },

    # ── 4. NHÓM RÁC NGUY HẠI & ĐIỆN TỬ (HAZARDOUS & E-WASTE) ──
    "battery": {
        "vn_name": "Pin Tiểu / Pin Li-ion / Pin Cúc Áo",
        "resin_code": "Rác Nguy Hại (Hazardous Waste - E-Waste)",
        "material": "Lithium, Chì, Thủy Ngân, Cadmium, Niken, Axit/Kiềm",
        "contamination_score": 100,
        "is_recyclable": False,
        "target_stream": "🔴 CÔ LẬP KHẨN CẤP — Thu Gom Nguy Hại Chuyên Sâu",
        "hazard_risk": "🚨 NGUY CƠ CHÁY NỔ & RÒ RỈ KIM LOẠI NẶNG ĐỘC HẠI",
        "hazard_level": "Extreme",
        "action": "⚠️ BÁO ĐỘNG ĐỎ! Tuyệt đối không ném vào thùng rác thường hay ép kiện. 1 viên pin Li-ion có thể gây cháy cả bãi rác."
    },
    "power bank": {
        "vn_name": "Sạc Dự Phòng / Pin Điện Thoại",
        "resin_code": "E-Waste Pin Lithium Polymer",
        "material": "Tế bào Li-Po kèm mạch điện tử vi điều khiển",
        "contamination_score": 95,
        "is_recyclable": False,
        "target_stream": "🔴 Nhà Máy Tái Chế Kim Loại Hiếm (Cobalt/Lithium)",
        "hazard_risk": "🚨 Nguy cơ nổ nhiệt (Thermal Runaway) khi bị va đập đè nén",
        "hazard_level": "Extreme",
        "action": "Cách ly trong thùng cát chống cháy, chuyển cơ sở xử lý rác thải điện tử."
    },
    "aerosol spray can": {
        "vn_name": "Bình Xịt Khí Nén / Bình Sơn / Bình Gas Mini",
        "resin_code": "Bình Chứa Áp Suất Khí Gas (Pressurized Can)",
        "material": "Thép/Nhôm chứa khí nén Hydrocarbon/Butane",
        "contamination_score": 90,
        "is_recyclable": False,
        "target_stream": "🔴 Cơ Sở Tiêu Hủy Bình Khí Áp Lực",
        "hazard_risk": "🚨 NGUY CƠ NỔ DỮ DỘI KHI MÁY ÉP ÉP NÁT",
        "hazard_level": "High",
        "action": "Không được đục thủng, cách ly khỏi khu vực nhiệt độ cao."
    },

    # ── 5. NHÓM RÁC Y TẾ & VỆ SINH (MEDICAL & SANITARY) ──
    "face mask": {
        "vn_name": "Khẩu Trang Y Tế Đã Qua Sử Dụng",
        "resin_code": "Rác Lây Nhiễm Sinh Học (Biohazard / Medical Waste)",
        "material": "Vải không dệt PP spunbond kèm màng lọc tĩnh điện meltblown",
        "contamination_score": 95,
        "is_recyclable": False,
        "target_stream": "⚪ Lò Đốt Rác Nhiệt Độ Cao (Thiêu Hủy Y Tế)",
        "hazard_risk": "⚠️ Nguy cơ lây nhiễm vi khuẩn, virus qua đường dịch tiết hô hấp",
        "hazard_level": "High",
        "action": "Cắt quai đeo để tránh bẫy động vật, bọc kín túi nilon đưa vào luồng rác đốt thiêu hủy."
    },
    "medical face mask": {
        "vn_name": "Khẩu Trang Y Tế",
        "resin_code": "Biohazard Material",
        "material": "Polypropylene không dệt + Nẹp kẽm mũi",
        "contamination_score": 95,
        "is_recyclable": False,
        "target_stream": "⚪ Lò Thiêu Đốt Rác Y Tế / Đốt RDF",
        "hazard_risk": "Rác lây nhiễm sinh học",
        "hazard_level": "High",
        "action": "Bỏ túi kín và khử khuẩn."
    },
    "pill blister pack": {
        "vn_name": "Vỉ Thuốc Tây Đã Hết",
        "resin_code": "Composite Màng PVC + Lá Nhôm Foil",
        "material": "Nhựa PVC dập phồng ép nhiệt với màng nhôm",
        "contamination_score": 75,
        "is_recyclable": False,
        "target_stream": "⚪ Đốt Rác Thu Hồi Năng Lượng (RDF)",
        "hazard_risk": "Tạp chất dược phẩm và màng kim loại bám chặt",
        "hazard_level": "Medium",
        "action": "Nếu còn thuốc thừa phải giao trả nhà thuốc; vỉ rỗng chuyển luồng rác sinh hoạt đốt."
    },
    "cigarette butt": {
        "vn_name": "Đầu Lọc Thuốc Lá",
        "resin_code": "Rác Độc Hại Vô Cơ (Cellulose Acetate)",
        "material": "Sợi nhựa nhân tạo chứa hắc ín, nicotine, asen, chì",
        "contamination_score": 100,
        "is_recyclable": False,
        "target_stream": "⚪ Thiêu Hủy Nhiệt An Toàn",
        "hazard_risk": "Nguy cơ tàn lửa gây cháy + 1 mẩu ô nhiễm 500L nước sạch",
        "hazard_level": "High",
        "action": "Dập tắt triệt để tàn lửa trước khi vứt. Tuyệt đối không vứt bừa bãi."
    }
}

# ── DANH SÁCH 32 PROMPTS RÁC ĐỜI THỰC MỞ RỘNG (DÀNH CHO YOLO-WORLD) ──
REAL_WORLD_WASTE_PROMPTS = [
    # Nhựa & Bao bì
    "plastic water bottle",
    "crushed beverage can",
    "disposable coffee cup",
    "plastic boba cup",
    "plastic straw",
    "plastic grocery bag",
    "bubble wrap",
    "styrofoam food container",
    "snack packaging bag",
    "instant noodle cup",
    # Kim loại & Giấy
    "aluminum can",
    "crushed can",
    "cardboard box",
    "greasy pizza box",
    "paper cup",
    "office paper",
    "tetra pak milk carton",
    # Nguy hại & Điện tử
    "battery",
    "power bank",
    "aerosol spray can",
    "electronic cable",
    "smartphone",
    "fluorescent bulb",
    "gas lighter",
    # Y tế & Đời sống
    "medical face mask",
    "pill blister pack",
    "rubber glove",
    "wet tissue wipe",
    "cigarette butt",
    # Hữu cơ đời thực
    "banana peel",
    "apple core",
    "food scraps"
]


class IndustrialAuditor:
    """Bộ Thẩm Định Độc Lập Vật Liệu và Chỉ Số Ô Nhiễm Rác Thải Công Nghiệp."""

    @staticmethod
    def audit_item(item_name: str, confidence: float = 0.5) -> Dict[str, Any]:
        """Thẩm định một vật thể được phát hiện và trả về hồ sơ công nghiệp toàn diện."""
        cleaned_key = item_name.strip().lower()

        # Tìm kiếm trực tiếp hoặc fuzzy match
        matched_info = None
        for k, v in INDUSTRIAL_KNOWLEDGE_BASE.items():
            if k == cleaned_key or k in cleaned_key or cleaned_key in k:
                matched_info = v
                break

        # Nếu không có trong từ điển chuyên sâu, dùng bộ suy luận mặc định
        if matched_info is None:
            matched_info = {
                "vn_name": item_name.capitalize(),
                "resin_code": "Vật liệu tổng hợp / Chưa chuẩn hóa",
                "material": "Đồ dùng sinh hoạt hỗn hợp",
                "contamination_score": 50,
                "is_recyclable": False,
                "target_stream": "⚪ Rác Thải Sinh Hoạt Còn Lại (Chôn Lấp / Đốt)",
                "hazard_risk": "Cần kiểm tra thủ công",
                "hazard_level": "Low",
                "action": "Phân loại sơ bộ vào rác vô cơ sinh hoạt."
            }

        score = matched_info["contamination_score"]
        is_hazard = matched_info["hazard_level"] in ["High", "Extreme"]

        # Đánh giá tiêu chuẩn tiếp nhận tái chế (Acceptance Criteria)
        if is_hazard:
            decision = "CÁCH LY KHẨN CẤP"
            badge_color = "#E74C3C" # Đỏ
        elif score < 30 and matched_info["is_recyclable"]:
            decision = "ĐỦ ĐIỀU KIỆN TÁI CHẾ"
            badge_color = "#2ECC71" # Xanh lá
        elif score >= 60 or not matched_info["is_recyclable"]:
            decision = "TỪ CHỐI TÁI CHẾ — CHUYỂN LÒ ĐỐT RDF"
            badge_color = "#F39C12" # Cam
        else:
            decision = "CẦN LÀM SẠCH TRƯỚC KHI TÁI CHẾ"
            badge_color = "#3498DB" # Xanh dương

        return {
            "name": item_name,
            "vn_name": matched_info["vn_name"],
            "confidence": confidence,
            "resin_code": matched_info["resin_code"],
            "material": matched_info["material"],
            "contamination_score": score,
            "is_recyclable": matched_info["is_recyclable"],
            "target_stream": matched_info["target_stream"],
            "hazard_risk": matched_info["hazard_risk"],
            "hazard_level": matched_info["hazard_level"],
            "is_hazard": is_hazard,
            "decision": decision,
            "badge_color": badge_color,
            "action": matched_info["action"]
        }

    @staticmethod
    def audit_batch(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Kiểm toán đồng loạt tất cả các vật thể trên một khung hình."""
        audit_reports = []
        has_critical_hazard = False
        recyclable_count = 0
        rdf_burn_count = 0
        hazard_count = 0

        for d in detections:
            report = IndustrialAuditor.audit_item(d["name"], d.get("confidence", 0.5))
            # Ghép tọa độ box nếu có
            if "box" in d:
                report["box"] = d["box"]
            audit_reports.append(report)

            if report["is_hazard"]:
                has_critical_hazard = True
                hazard_count += 1
            elif report["is_recyclable"] and report["contamination_score"] < 50:
                recyclable_count += 1
            else:
                rdf_burn_count += 1

        return {
            "items": audit_reports,
            "total_items": len(audit_reports),
            "has_critical_hazard": has_critical_hazard,
            "summary": {
                "recyclable": recyclable_count,
                "rdf_waste_to_energy": rdf_burn_count,
                "hazard_isolated": hazard_count
            }
        }
