import cv2
import numpy as np
import base64
import json
import math
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')

DP_DENSITY = 3  # 1dp = 3px (xxhdpi)
MIN_COMPONENT_AREA = 80
MIN_COMPONENT_WH = 12

# ─── 字体加载 ─────────────────────────────────────────────

def load_font(size=14):
    """加载支持中文的系统字体"""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()

FONT_SM = load_font(13)
FONT_MD = load_font(15)
FONT_LG = load_font(18)
FONT_XS = load_font(11)


def pil_text_size(text, font):
    """兼容 Pillow 的 text size 计算"""
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


def cv2_to_base64(img):
    success, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8') if success else None


def resize_to_match(impl, design_shape):
    dh, dw = design_shape[:2]
    if impl.shape[:2] != (dh, dw):
        return cv2.resize(impl, (dw, dh), interpolation=cv2.INTER_CUBIC)
    return impl


def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    from skimage.metrics import structural_similarity as ssim
    score, _ = ssim(gray1, gray2, full=True)
    return round(float(score), 4)


# ─── 元素检测 ─────────────────────────────────────────────

def detect_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_COMPONENT_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_COMPONENT_WH or h < MIN_COMPONENT_WH:
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes


def merge_overlapping_boxes(boxes, iou_thresh=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = [boxes[0]]
    for box in boxes[1:]:
        prev = merged[-1]
        ix1, iy1 = max(prev[0], box[0]), max(prev[1], box[1])
        ix2, iy2 = min(prev[2], box[2]), min(prev[3], box[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (prev[2] - prev[0]) * (prev[3] - prev[1])
        a2 = (box[2] - box[0]) * (box[3] - box[1])
        iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0
        if iou > iou_thresh:
            merged[-1] = [min(prev[0], box[0]), min(prev[1], box[1]),
                          max(prev[2], box[2]), max(prev[3], box[3])]
        else:
            merged.append(box)
    return merged


# ─── 视觉属性比较 ─────────────────────────────────────────

def compare_colors(design_roi, impl_roi):
    """比较两个区域的平均颜色，返回差异描述"""
    if design_roi.size == 0 or impl_roi.size == 0:
        return None

    d_mean = np.mean(design_roi, axis=(0, 1))  # BGR
    i_mean = np.mean(impl_roi, axis=(0, 1))
    diff = np.linalg.norm(d_mean - i_mean)

    if diff < 15:
        return None  # 颜色基本一致

    d_rgb = [int(d_mean[2]), int(d_mean[1]), int(d_mean[0])]
    i_rgb = [int(i_mean[2]), int(i_mean[1]), int(i_mean[0])]

    # 亮度判断
    d_lum = 0.299 * d_rgb[0] + 0.587 * d_rgb[1] + 0.114 * d_rgb[2]
    i_lum = 0.299 * i_rgb[0] + 0.587 * i_rgb[1] + 0.114 * i_rgb[2]

    return {
        "design_rgb": d_rgb,
        "impl_rgb": i_rgb,
        "diff_norm": round(float(diff), 1),
        "direction": "偏深" if i_lum < d_lum else "偏浅" if diff > 30 else "色差",
    }


def estimate_border_radius(img_roi):
    """估算圆角半径 (px)"""
    if img_roi is None or img_roi.size == 0:
        return 0
    h, w = img_roi.shape[:2]
    if h < 6 or w < 6:
        return 0

    gray = img_roi if len(img_roi.shape) == 2 else cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    corner_size = min(h, w, 30)

    corners = [
        binary[0:corner_size, 0:corner_size],
        binary[0:corner_size, w - corner_size:w],
        binary[h - corner_size:h, 0:corner_size],
        binary[h - corner_size:h, w - corner_size:w]
    ]
    radii = []
    for corner in corners:
        rows = np.any(corner > 0, axis=1)
        cols = np.any(corner > 0, axis=0)
        r = np.argmax(rows) if np.any(rows) else 0
        c = np.argmax(cols) if np.any(cols) else 0
        radii.append(max(r, c))
    return int(np.median(radii))


def compare_border_radius(design_roi, impl_roi):
    """比较圆角差异"""
    d_r = estimate_border_radius(design_roi)
    i_r = estimate_border_radius(impl_roi)
    diff_px = abs(d_r - i_r)
    diff_dp = round(diff_px / DP_DENSITY, 1)

    if diff_px <= 2:
        return None  # 圆角基本一致

    if i_r < d_r:
        desc = f"圆角偏小 {diff_dp}dp (设计{d_r}px→实现{i_r}px)"
    elif i_r > d_r:
        desc = f"圆角偏大 {diff_dp}dp (设计{d_r}px→实现{i_r}px)"
    else:
        return None

    return {
        "design_px": d_r,
        "impl_px": i_r,
        "diff_px": diff_px,
        "diff_dp": diff_dp,
        "description": desc,
    }


def compare_spacing(design_boxes):
    """计算设计稿中相邻元素的间距"""
    spacings = []
    for i in range(len(design_boxes)):
        for j in range(i + 1, len(design_boxes)):
            a, b = design_boxes[i], design_boxes[j]
            # 水平间距
            if a[3] >= b[1] and a[1] <= b[3]:  # 垂直有重叠
                if a[2] <= b[0]:
                    gap = b[0] - a[2]
                    spacings.append({"pair": [i, j], "axis": "h", "dp": round(gap / DP_DENSITY, 1)})
                elif b[2] <= a[0]:
                    gap = a[0] - b[2]
                    spacings.append({"pair": [i, j], "axis": "h", "dp": round(gap / DP_DENSITY, 1)})
            # 垂直间距
            if a[2] >= b[0] and a[0] <= b[2]:  # 水平有重叠
                if a[3] <= b[1]:
                    gap = b[1] - a[3]
                    spacings.append({"pair": [i, j], "axis": "v", "dp": round(gap / DP_DENSITY, 1)})
                elif b[3] <= a[1]:
                    gap = a[1] - b[3]
                    spacings.append({"pair": [i, j], "axis": "v", "dp": round(gap / DP_DENSITY, 1)})
    return spacings


def detect_shadow(img_roi):
    """检测阴影"""
    if img_roi.size == 0:
        return None
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    # 检查边缘是否有渐变（阴影特征）
    h, w = gray.shape
    if h < 10 or w < 10:
        return None
    # 检查底部和右侧边缘的亮度梯度
    bottom_strip = gray[max(0, h - 5):h, :]
    right_strip = gray[:, max(0, w - 5):w]
    bottom_var = np.std(bottom_strip)
    right_var = np.std(right_strip)
    if bottom_var > 15 or right_var > 15:
        return "has_shadow"
    return None


# ─── 元素配对 ─────────────────────────────────────────────

def match_components(design_boxes, impl_boxes, design_img, impl_img):
    results = []
    used_impl = set()

    for d_idx, db in enumerate(design_boxes):
        best_match = None
        best_score = -1

        dcx = (db[0] + db[2]) / 2
        dcy = (db[1] + db[3]) / 2
        dw = db[2] - db[0]
        dh = db[3] - db[1]
        d_area = dw * dh

        for i_idx, ib in enumerate(impl_boxes):
            if i_idx in used_impl:
                continue

            icx = (ib[0] + ib[2]) / 2
            icy = (ib[1] + ib[3]) / 2
            iw = ib[2] - ib[0]
            ih = ib[3] - ib[1]

            w_ratio = min(dw, iw) / max(dw, iw) if max(dw, iw) > 0 else 0
            h_ratio = min(dh, ih) / max(dh, ih) if max(dh, ih) > 0 else 0
            size_sim = (w_ratio + h_ratio) / 2
            dist = math.sqrt((dcx - icx) ** 2 + (dcy - icy) ** 2)

            ix1 = max(db[0], ib[0])
            iy1 = max(db[1], ib[1])
            ix2 = min(db[2], ib[2])
            iy2 = min(db[3], ib[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            i_area = iw * ih
            union = d_area + i_area - inter
            iou = inter / union if union > 0 else 0

            score = iou * 100 + size_sim * 10 - dist * 0.01
            if score > best_score:
                best_score = score
                best_match = i_idx

        if best_match is not None and best_score > 1:
            ib = impl_boxes[best_match]
            used_impl.add(best_match)

            dx_px = ((ib[0] + ib[2]) / 2) - ((db[0] + db[2]) / 2)
            dy_px = ((ib[1] + ib[3]) / 2) - ((db[1] + db[3]) / 2)
            iw, ih = ib[2] - ib[0], ib[3] - ib[1]

            # 提取 ROI 做视觉比较
            dh_img, dw_img = design_img.shape[:2]
            d_roi = design_img[max(0, db[1]):min(dh_img, db[3]), max(0, db[0]):min(dw_img, db[2])]
            i_roi = impl_img[max(0, ib[1]):min(dh_img, ib[3]), max(0, ib[0]):min(dw_img, ib[2])]

            color_cmp = compare_colors(d_roi, i_roi)
            radius_cmp = compare_border_radius(d_roi, i_roi)

            # 收集所有差异
            diffs = []

            # 位置
            dx_dp = round(dx_px / DP_DENSITY, 1)
            dy_dp = round(dy_px / DP_DENSITY, 1)
            if abs(dx_dp) >= 0.3 or abs(dy_dp) >= 0.3:
                pos_parts = []
                if abs(dx_dp) >= 0.3:
                    pos_parts.append(f"{'右' if dx_dp > 0 else '左'}偏移{abs(dx_dp)}dp")
                if abs(dy_dp) >= 0.3:
                    pos_parts.append(f"{'下' if dy_dp > 0 else '上'}偏移{abs(dy_dp)}dp")
                diffs.append({"type": "position", "description": "，".join(pos_parts),
                              "dx_dp": dx_dp, "dy_dp": dy_dp})

            # 尺寸
            dw_dp = round((iw - dw) / DP_DENSITY, 1)
            dh_dp = round((ih - dh) / DP_DENSITY, 1)
            if abs(dw_dp) >= 0.3 or abs(dh_dp) >= 0.3:
                size_parts = []
                if abs(dw_dp) >= 0.3:
                    size_parts.append(f"宽{'+' if dw_dp > 0 else ''}{dw_dp}dp")
                if abs(dh_dp) >= 0.3:
                    size_parts.append(f"高{'+' if dh_dp > 0 else ''}{dh_dp}dp")
                diffs.append({"type": "size", "description": "，".join(size_parts),
                              "dw_dp": dw_dp, "dh_dp": dh_dp})

            # 颜色
            if color_cmp:
                diffs.append({"type": "color", "description": color_cmp["direction"],
                              "design_rgb": color_cmp["design_rgb"],
                              "impl_rgb": color_cmp["impl_rgb"],
                              "diff_norm": color_cmp["diff_norm"]})

            # 圆角
            if radius_cmp:
                diffs.append({"type": "border-radius", "description": radius_cmp["description"],
                              "design_px": radius_cmp["design_px"],
                              "impl_px": radius_cmp["impl_px"],
                              "diff_dp": radius_cmp["diff_dp"]})

            results.append({
                "design_bbox": list(db),
                "impl_bbox": list(ib),
                "dx_dp": dx_dp,
                "dy_dp": dy_dp,
                "dw_dp": dw_dp,
                "dh_dp": dh_dp,
                "diffs": diffs,
                "has_diff": len(diffs) > 0,
            })

    # 未匹配的实现元素
    for i_idx, ib in enumerate(impl_boxes):
        if i_idx not in used_impl:
            results.append({
                "design_bbox": None,
                "impl_bbox": list(ib),
                "dx_dp": None, "dy_dp": None, "dw_dp": None, "dh_dp": None,
                "diffs": [{"type": "extra", "description": "实现中有但设计稿未找到"}],
                "has_diff": True,
                "unmatched": "impl_only",
            })

    # 未匹配的设计元素
    used_design = set()
    for r in results:
        b = r.get("design_bbox")
        if b:
            used_design.add(tuple(b))
    for d_idx, db in enumerate(design_boxes):
        if tuple(db) not in used_design:
            results.append({
                "design_bbox": list(db),
                "impl_bbox": None,
                "dx_dp": None, "dy_dp": None, "dw_dp": None, "dh_dp": None,
                "diffs": [{"type": "missing", "description": "设计稿有但实现中缺失"}],
                "has_diff": True,
                "unmatched": "design_only",
            })

    return results


# ─── 严重度 & 描述 ───────────────────────────────────────

def classify_severity(item):
    if item.get("unmatched"):
        return "critical"

    dx = abs(item.get("dx_dp") or 0)
    dy = abs(item.get("dy_dp") or 0)
    dw = abs(item.get("dw_dp") or 0)
    dh = abs(item.get("dh_dp") or 0)

    # 检查颜色/圆角差异
    has_color = any(d["type"] == "color" for d in item.get("diffs", []))
    has_radius = any(d["type"] == "border-radius" for d in item.get("diffs", []))

    max_offset = max(dx, dy, dw, dh)

    if max_offset >= 8 or (has_color and max_offset >= 3):
        return "critical"
    elif max_offset >= 3 or has_color or has_radius:
        return "major"
    elif max_offset >= 0.5:
        return "minor"
    else:
        return "ok"


def build_description(diffs):
    """从差异列表构建描述"""
    if not diffs:
        return "无差异 ✅"
    parts = [d["description"] for d in diffs]
    return " | ".join(parts)


# ─── 标注图生成 ───────────────────────────────────────────

def generate_annotated_image(design, modules):
    """用 PIL 渲染标注图，支持中文"""
    h, w = design.shape[:2]
    color_map = {
        "critical": (255, 0, 0),
        "major": (255, 165, 0),
        "minor": (255, 200, 0),
        "ok": (0, 200, 0),
    }
    bgr_to_rgb = lambda c: (c[2], c[1], c[0])

    # 右侧面板宽度
    panel_w = max(300, int(w * 0.45))
    ext_w = w + panel_w

    # 用 PIL 创建画布
    canvas = Image.new("RGB", (ext_w, h), (255, 255, 255))
    # 左侧贴设计稿
    design_pil = cv2_to_pil(design)
    canvas.paste(design_pil, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # 面板标题
    draw.text((w + 14, 10), "📐 设计走查报告", fill=(50, 50, 50), font=FONT_LG)
    draw.line([(w + 10, 36), (ext_w - 10, 36)], fill=(200, 200, 200), width=1)

    items = [(i, m) for i, m in enumerate(modules) if m["severity"] != "ok"]
    panel_y = 46
    line_h = 20

    for idx, (orig_i, m) in enumerate(items):
        bbox = m.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        sev = m["severity"]
        color = color_map.get(sev, (255, 255, 0))
        mid = m["id"]

        # ── 左侧画框 + 编号 ──
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"#{mid}"
        lw, lh = pil_text_size(label, FONT_SM)
        draw.rectangle([x1, y1 - lh - 6, x1 + lw + 10, y1], fill=color)
        draw.text((x1 + 5, y1 - lh - 4), label, fill=(255, 255, 255), font=FONT_SM)

        # 偏移箭头（画在 PIL 上）
        dx = m.get("dx_dp") or 0
        dy = m.get("dy_dp") or 0
        if abs(dx) >= 0.3 or abs(dy) >= 0.3:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            al = 30
            ex = cx + int(dx / max(abs(dx), 0.1) * al)
            ey = cy + int(dy / max(abs(dy), 0.1) * al)
            draw.line([(cx, cy), (ex, ey)], fill=color, width=2)
            # 箭头三角
            angle = math.atan2(ey - cy, ex - cx)
            for da in [2.5, -2.5]:
                ax = ex - int(10 * math.cos(angle + da))
                ay = ey - int(10 * math.sin(angle + da))
                draw.line([(ex, ey), (ax, ay)], fill=color, width=2)

        # ── 右侧面板 ──
        sev_cn = {"critical": "严重", "major": "主要", "minor": "轻微"}.get(sev, sev)
        sev_color = color
        draw.text((w + 14, panel_y), f"#{mid}", fill=sev_color, font=FONT_MD)
        tw, _ = pil_text_size(f"#{mid}", FONT_MD)
        draw.text((w + 14 + tw + 6, panel_y + 1), f"[{sev_cn}]", fill=sev_color, font=FONT_XS)
        panel_y += line_h + 2

        # 每条差异
        type_label = {"position": "位置", "size": "尺寸", "color": "颜色",
                       "border-radius": "圆角", "extra": "多余", "missing": "缺失"}
        for d in m.get("diffs", []):
            t = d["type"]
            desc = d.get("description", "")
            icon = {"position": "📍", "size": "📐", "color": "🎨",
                     "border-radius": "⬜", "extra": "➕", "missing": "❌"}.get(t, "⚠️")
            prefix = f"{icon} {type_label.get(t, t)}："

            draw.text((w + 24, panel_y), prefix, fill=(100, 100, 100), font=FONT_XS)
            pw, _ = pil_text_size(prefix, FONT_XS)
            draw.text((w + 24 + pw, panel_y), desc, fill=(60, 60, 60), font=FONT_XS)
            panel_y += line_h - 4

            # 颜色差异显示色块
            if t == "color" and d.get("design_rgb") and d.get("impl_rgb"):
                dr, dg, db_c = d["design_rgb"]
                ir, ig, ib_c = d["impl_rgb"]
                bx = w + 34
                draw.rectangle([bx, panel_y, bx + 20, panel_y + 14], fill=(dr, dg, db_c),
                               outline=(180, 180, 180), width=1)
                draw.text((bx + 24, panel_y), "设计", fill=(120, 120, 120), font=FONT_XS)
                draw.rectangle([bx + 80, panel_y, bx + 100, panel_y + 14], fill=(ir, ig, ib_c),
                               outline=(180, 180, 180), width=1)
                draw.text((bx + 104, panel_y), "实现", fill=(120, 120, 120), font=FONT_XS)
                panel_y += line_h

        panel_y += 8

    # 底部图例
    panel_y = max(panel_y + 10, h - 50)
    draw.line([(w + 10, panel_y), (ext_w - 10, panel_y)], fill=(200, 200, 200), width=1)
    panel_y += 8
    legends = [("📍", "位置偏移"), ("📐", "尺寸差异"), ("🎨", "颜色差异"), ("⬜", "圆角差异")]
    for i, (icon, label) in enumerate(legends):
        lx = w + 14 + i * 130
        draw.text((lx, panel_y), f"{icon} {label}", fill=(140, 140, 140), font=FONT_XS)

    # 转回 OpenCV 格式给 base64
    result = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    return result


# ─── API ──────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        return _analyze_impl()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "trace": tb}), 500

def _analyze_impl():
    if 'design' not in request.files or 'screenshot' not in request.files:
        return jsonify({"error": "请上传设计稿和实现截图"}), 400

    sensitivity = int(request.form.get('sensitivity', 30))

    design_bytes = np.frombuffer(request.files['design'].read(), np.uint8)
    impl_bytes = np.frombuffer(request.files['screenshot'].read(), np.uint8)
    design = cv2.imdecode(design_bytes, cv2.IMREAD_COLOR)
    impl = cv2.imdecode(impl_bytes, cv2.IMREAD_COLOR)

    if design is None or impl is None:
        return jsonify({"error": "无法读取图片文件"}), 400

    impl = resize_to_match(impl, design.shape)
    ssim_score = compute_ssim(design, impl)

    # 检测 + 配对
    design_boxes = merge_overlapping_boxes(detect_contours(design))
    impl_boxes = merge_overlapping_boxes(detect_contours(impl))
    matched = match_components(design_boxes, impl_boxes, design, impl)

    # 构建模块输出
    modules = []
    for i, item in enumerate(matched):
        bbox = item.get("design_bbox") or item.get("impl_bbox")
        sev = classify_severity(item)
        desc = build_description(item.get("diffs", []))

        modules.append({
            "id": i + 1,
            "bbox": bbox,
            "severity": sev,
            "description": desc,
            "dx_dp": item.get("dx_dp"),
            "dy_dp": item.get("dy_dp"),
            "dw_dp": item.get("dw_dp"),
            "dh_dp": item.get("dh_dp"),
            "diffs": item.get("diffs", []),
            "design_bbox": item.get("design_bbox"),
            "impl_bbox": item.get("impl_bbox"),
        })

    # 标注图
    annotated = generate_annotated_image(design, modules)

    # 统计
    critical = sum(1 for m in modules if m["severity"] == "critical")
    major = sum(1 for m in modules if m["severity"] == "major")
    minor = sum(1 for m in modules if m["severity"] == "minor")
    ok_count = sum(1 for m in modules if m["severity"] == "ok")
    issues = [m for m in modules if m["severity"] != "ok"]

    verdict = "需要修改" if (critical > 0 or major > 0) else "验收通过"

    # 摘要
    summary_parts = [f"检测 {len(design_boxes)} 个设计元素 / {len(impl_boxes)} 个实现元素"]
    if issues:
        cats = {}
        for m in issues:
            for d in m.get("diffs", []):
                cats[d["type"]] = cats.get(d["type"], 0) + 1
        cat_desc = []
        type_labels = {"position": "位置偏移", "size": "尺寸差异", "color": "颜色差异",
                        "border-radius": "圆角差异", "extra": "多余元素", "missing": "缺失元素"}
        for t, c in sorted(cats.items(), key=lambda x: -x[1]):
            cat_desc.append(f"{type_labels.get(t, t)} {c}个")
        summary_parts.append("问题类型：" + "，".join(cat_desc))
    else:
        summary_parts.append("所有维度均无显著差异")

    # Top 5
    def module_max_offset(m):
        return max(abs(m.get("dx_dp") or 0), abs(m.get("dy_dp") or 0),
                   abs(m.get("dw_dp") or 0), abs(m.get("dh_dp") or 0))
    top = sorted(issues, key=module_max_offset, reverse=True)[:5]

    return jsonify({
        "ssim": ssim_score,
        "design_size": [int(design.shape[1]), int(design.shape[0])],
        "modules": modules,
        "annotated_image": cv2_to_base64(annotated),
        "design_image": cv2_to_base64(design),
        "impl_image": cv2_to_base64(impl),
        "verdict": verdict,
        "summary": "；".join(summary_parts),
        "top_offsets": [{
            "id": m["id"],
            "description": m["description"],
            "severity": m["severity"],
            "dx_dp": m.get("dx_dp"),
            "dy_dp": m.get("dy_dp"),
            "diffs": m.get("diffs", []),
        } for m in top],
    })


@app.route('/api/export', methods=['POST'])
def export():
    data = request.get_json()
    if not data or 'annotated_image' not in data:
        return jsonify({"error": "缺少标注图片数据"}), 400
    img_bytes = base64.b64decode(data['annotated_image'])
    return img_bytes, 200, {
        'Content-Type': 'image/png',
        'Content-Disposition': 'attachment; filename=design-check-report.png'
    }


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5007, debug=True)
