import time
import cv2
import datetime
import numpy as np
from .central_logger import get_logger

# --- Information About Script ---
__name__ = "Utilities for transformsai-core"
__author__ = "TransformsAI"

# Module-level logger
_logger = get_logger(__name__)

# Cache for color palettes to avoid regeneration (saves CPU on repeated calls)
_COLOR_PALETTE_CACHE = {}

UPLOAD_IMAGE_MAX_WIDTH_DEFAULT = 1920
JPEG_DEFAULT_QUALITY = 65

def time_to_string(input):
    time_tuple = time.gmtime(input)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time_tuple)

def resize_frame(frame, max_width=UPLOAD_IMAGE_MAX_WIDTH_DEFAULT):
    """
    Resizes an OpenCV frame to a maximum width while maintaining aspect ratio.

    Args:
        frame (np.ndarray): The image frame to resize.
        max_width (int): The maximum desired width. If the frame's width is
                         already less than or equal to max_width, the original
                         frame is returned.

    Returns:
        np.ndarray: The resized frame.
    """
    height, width = frame.shape[:2]

    if width > max_width:
        # Calculate the ratio
        ratio = max_width / width
        # Calculate new dimensions
        new_width = max_width
        new_height = int(height * ratio)
        # Resize the image using INTER_AREA for shrinking
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame
    else:
        # No resizing needed
        return frame

def mat_to_response(frame, max_width=UPLOAD_IMAGE_MAX_WIDTH_DEFAULT, jpeg_quality=JPEG_DEFAULT_QUALITY, filename="image.jpg", timestamp=None, add_timestamp=False):
    """
    Resizes (if necessary) and encodes an OpenCV frame (NumPy array)
    to JPEG bytes in memory with a specified quality.

    Args:
        frame (np.ndarray): The image frame to encode.
        max_width (int): Maximum width for resizing before encoding.
        jpeg_quality (int): JPEG compression quality (0-100).
        timestamp (float, optional): Timestamp to be added to the image. Current time if None.
        add_timestamp (bool): If True, add timestamp overlay to the frame.

    Returns:
        tuple | None: A tuple suitable for the 'files' parameter in requests
                      (filename, image_bytes, content_type), or None if encoding fails.
    """
    try:
        # 0. Replace timestamp (modifies frame inplace for efficiency)
        if add_timestamp:
            if timestamp is None:
                timestamp = time.time()
            frame = hide_camera_timestamp_and_add_current_time(frame, timestamp=timestamp, inplace=True)
        
        # 1. Resize the frame
        resized_frame = resize_frame(frame, max_width)

        # 2. Encode the resized image to JPEG format in memory
        # Use tuple for encode_params (immutable, no allocation churn)
        encode_params = (int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality)
        ret, encoded_image = cv2.imencode(".jpg", resized_frame, encode_params)

        if not ret:
            _logger.error("Could not encode image")
            return None

        # Convert the encoded image NumPy array to bytes
        image_bytes = encoded_image.tobytes()

        # Prepare the tuple for sending (filename, content_bytes, content_type)
        return (filename, image_bytes, "image/jpeg")
    except Exception as e:
        _logger.error(f"Error during image resizing or encoding: {e}")
        return None

# --- styled stream badges (variant B: slate + indigo accent bar + live dot) ---
# PIL patches are pre-rendered and cached (location: once; time: once per
# second) and numpy-blended per frame, so the per-frame cost at 10 fps is a
# couple of small array ops. Falls back to the cv2 badge if PIL/fonts fail.
_BADGE_CACHE = {}
_BADGE_BG_RGBA = (15, 23, 42, 180)        # slate
_BADGE_ACCENT_RGBA = (99, 102, 241, 230)  # indigo bar
_BADGE_DOT_RGB = (74, 222, 128)           # live dot
_BADGE_KICKER_RGB = (147, 197, 253)       # "AI SURVEILLANCE"
_FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


def _badge_font(bold, px):
    from PIL import ImageFont
    path = _FONT_DIR + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    key = ("font", path, px)
    f = _BADGE_CACHE.get(key)
    if f is None:
        f = ImageFont.truetype(path, px)
        _BADGE_CACHE[key] = f
    return f


def _badge_patch(lines, pad_x, pad_y, gap, radius, dot_px=0):
    """Render a rounded slate badge with an accent bar as (bgr, alpha) arrays.

    lines: list of (text, PIL_font, rgb) drawn top to bottom, centered when
    multiple, left-aligned after the optional live dot when single.
    """
    from PIL import Image, ImageDraw
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    boxes = [d0.textbbox((0, 0), t, font=f) for t, f, _ in lines]
    ws = [b[2] - b[0] for b in boxes]
    hs = [b[3] - b[1] for b in boxes]
    dot_w = dot_px + max(6, dot_px // 1) if dot_px else 0
    bw = max(ws) + 2 * pad_x + dot_w
    bh = sum(hs) + gap * (len(lines) - 1) + 2 * pad_y
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, bw - 1, bh - 1), radius=radius, fill=_BADGE_BG_RGBA)
    d.rounded_rectangle((0, 0, 4, bh - 1), radius=2, fill=_BADGE_ACCENT_RGBA)
    if dot_px:
        cy = bh // 2
        d.ellipse((pad_x, cy - dot_px // 2, pad_x + dot_px, cy + dot_px // 2),
                  fill=_BADGE_DOT_RGB + (255,))
    y = pad_y
    for (t, f, rgb), b, w, h in zip(lines, boxes, ws, hs):
        x = pad_x + dot_w if len(lines) == 1 else (bw - w) // 2
        d.text((x, y - b[1]), t, font=f, fill=rgb + (255,))
        y += h + gap
    arr = np.asarray(img, dtype=np.uint8)
    bgr = arr[..., 2::-1].copy()
    alpha = (arr[..., 3:4].astype(np.float32)) / 255.0
    return bgr, alpha


def _blend_patch(frame, patch, x0, y0):
    bgr, alpha = patch
    h, w = bgr.shape[:2]
    fh, fw = frame.shape[:2]
    x0 = max(0, min(x0, fw - w)); y0 = max(0, min(y0, fh - h))
    roi = frame[y0:y0 + h, x0:x0 + w]
    roi[:] = (bgr.astype(np.float32) * alpha
              + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def _draw_badges_pil(frame, time_string, corner_label, margin_x, margin_y):
    fh, fw = frame.shape[:2]
    s = fh / 720.0
    key_t = ("time", time_string, fw, fh)
    patch_t = _BADGE_CACHE.get(key_t)
    if patch_t is None:
        # keep the cache from accumulating one entry per second
        for k in [k for k in _BADGE_CACHE if k[0] == "time"]:
            del _BADGE_CACHE[k]
        patch_t = _badge_patch(
            [(time_string, _badge_font(True, max(12, int(19 * s))), (255, 255, 255))],
            pad_x=int(14 * s), pad_y=int(9 * s), gap=0,
            radius=max(4, int(10 * s)), dot_px=max(6, int(8 * s)))
        _BADGE_CACHE[key_t] = patch_t
    _blend_patch(frame, patch_t, margin_x, margin_y)

    if corner_label:
        key_l = ("loc", str(corner_label), fw, fh)
        patch_l = _BADGE_CACHE.get(key_l)
        if patch_l is None:
            kicker = " ".join("AI SURVEILLANCE")
            patch_l = _badge_patch(
                [(kicker, _badge_font(False, max(8, int(12 * s))), _BADGE_KICKER_RGB),
                 (str(corner_label), _badge_font(True, max(13, int(21 * s))),
                  (255, 255, 255))],
                pad_x=int(16 * s), pad_y=int(10 * s), gap=int(5 * s),
                radius=max(4, int(10 * s)))
            _BADGE_CACHE[key_l] = patch_l
        h, w = patch_l[0].shape[:2]
        _blend_patch(frame, patch_l, fw - margin_x - w, fh - margin_y - h)
    return frame


def hide_camera_timestamp_and_add_current_time(
    frame,
    camera_ts_rect_coords=None,
    camera_ts_rect_ratios=(0.015, 0.05, 0.25, 0.035), 
    hide_rect_color=(255, 255, 255),
    new_ts_position_on_rect=True,
    new_ts_custom_position=None,
    new_ts_font=cv2.FONT_HERSHEY_SIMPLEX,
    new_ts_font_scale=None,
    new_ts_font_color=(0, 0, 0),
    new_ts_font_thickness=1,
    new_ts_padding_ratio=0.1,
    timestamp=None,
    inplace=False,
    time_format="%Y-%m-%d %H:%M:%S",
    badge=True,
    badge_alpha=0.55,
    corner_label=None,
):
    """
    Hides a region on a frame (defined by pixel coordinates or ratios)
    with a rectangle and adds the current system time (centered) onto that rectangle
    or at a custom position.

    Args:
        frame (np.ndarray): The input video frame (OpenCV BGR format).
        camera_ts_rect_coords (tuple, optional): (x, y, w, h) in pixels. Overrides ratios.
        camera_ts_rect_ratios (tuple, optional): (x_r, y_r, w_r, h_r) ratios (0.0-1.0).
                                       Default: (0.015, 0.05, 0.26, 0.035).
        hide_rect_color (tuple, optional): BGR color of hiding rectangle. Default: white.
        new_ts_position_on_rect (bool, optional): True to place new TS on hiding rect. Default: True.
        new_ts_custom_position (tuple, optional): (x,y) for new TS if not on rect. Default: None.
        new_ts_font (int, optional): Font type. Default: cv2.FONT_HERSHEY_SIMPLEX.
        new_ts_font_scale (float, optional): Font scale. If None, auto-calculated. Default: None.
        new_ts_font_color (tuple, optional): BGR color for new TS. Default: black.
        new_ts_font_thickness (int, optional): Thickness for new TS. Default: 1.
        new_ts_padding_ratio (float, optional): Padding for new TS within hiding rect,
                                                as a ratio of the rectangle's smaller dimension.
                                                Default: 0.1.
        timestamp (float, optional): Unix timestamp to display. Uses current time if None.
        inplace (bool, optional): If True, modify frame in-place (saves ~6MB for 1080p).
                                  Default: False (backward compatible).
        time_format (str, optional): strftime format for the rendered timestamp.
                                     Default: "%Y-%m-%d %H:%M:%S".

    Returns:
        np.ndarray: The modified frame.
    """
    # Optimize for edge devices: avoid copy if caller allows inplace modification
    output_frame = frame if inplace else frame.copy()
    frame_h, frame_w = output_frame.shape[:2]

    # 1. Determine the rectangle coordinates (pixels)
    if camera_ts_rect_coords:
        rect_x, rect_y, rect_w, rect_h = camera_ts_rect_coords
    elif camera_ts_rect_ratios:
        xr, yr, wr, hr = camera_ts_rect_ratios
        rect_x = int(frame_w * xr)
        rect_y = int(frame_h * yr)
        rect_w = int(frame_w * wr)
        rect_h = int(frame_h * hr)
        rect_w = max(1, rect_w)
        rect_h = max(1, rect_h)
    else:
        print("Warning: No rectangle coordinates or ratios. Using small default.")
        rect_x, rect_y, rect_w, rect_h = 10, 10, 100, 20

    # 3. Get current time and format it
    current_unix_time = time.time() if timestamp is None else timestamp
    dt_object = datetime.datetime.fromtimestamp(current_unix_time)
    time_string = dt_object.strftime(time_format)

    # Badge mode: styled slate badges (PIL, cached patches). Assumes the
    # camera's own OSD is disabled (nothing left to hide). Falls back to the
    # plain cv2 pill below if PIL or the DejaVu fonts are unavailable.
    if badge and _BADGE_CACHE.get("pil_broken") is not True:
        try:
            return _draw_badges_pil(output_frame, time_string, corner_label,
                                    max(10, rect_x), max(10, rect_y))
        except Exception:  # noqa: BLE001 -- never let styling kill the capture thread
            _BADGE_CACHE["pil_broken"] = True

    # cv2 fallback: auto-sized semi-transparent rounded pill
    if badge:
        scale = np.clip(rect_h * 0.022, 0.3, 2.5) if new_ts_font_scale is None \
            else new_ts_font_scale
        thick = max(1, int(scale + 0.5))
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), baseline = cv2.getTextSize(time_string, font, scale, thick)
        pad = max(4, int(th * 0.6))
        bw, bh = tw + 2 * pad, th + baseline + 2 * pad
        x0 = int(np.clip(rect_x, 0, max(0, frame_w - bw)))
        y0 = int(np.clip(rect_y, 0, max(0, frame_h - bh)))
        x1, y1 = min(x0 + bw, frame_w), min(y0 + bh, frame_h)
        r = max(2, min(bh // 3, 12))

        roi = output_frame[y0:y1, x0:x1]
        ov = roi.copy()
        w, h = x1 - x0, y1 - y0
        color = tuple(int(c) for c in hide_rect_color)
        cv2.rectangle(ov, (r, 0), (w - r, h), color, -1)
        cv2.rectangle(ov, (0, r), (w, h - r), color, -1)
        for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
            cv2.circle(ov, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, badge_alpha, roi, 1.0 - badge_alpha, 0, dst=roi)

        cv2.putText(output_frame, time_string, (x0 + pad, y0 + pad + th),
                    font, scale, tuple(int(c) for c in new_ts_font_color),
                    thick, cv2.LINE_AA)

        # Optional location/name badge, bottom-right, same style as the time badge
        if corner_label:
            (lw, lh), lb = cv2.getTextSize(str(corner_label), font, scale, thick)
            lpad = max(4, int(lh * 0.6))
            lbw, lbh = lw + 2 * lpad, lh + lb + 2 * lpad
            margin = max(6, rect_x)  # mirror the time badge's left margin
            lx0 = max(0, frame_w - margin - lbw)
            ly0 = max(0, frame_h - margin - lbh)
            lx1, ly1 = min(lx0 + lbw, frame_w), min(ly0 + lbh, frame_h)
            lr = max(2, min((ly1 - ly0) // 3, 12))
            lroi = output_frame[ly0:ly1, lx0:lx1]
            lov = lroi.copy()
            w2, h2 = lx1 - lx0, ly1 - ly0
            color = tuple(int(c) for c in hide_rect_color)
            cv2.rectangle(lov, (lr, 0), (w2 - lr, h2), color, -1)
            cv2.rectangle(lov, (0, lr), (w2, h2 - lr), color, -1)
            for cx, cy in ((lr, lr), (w2 - lr, lr), (lr, h2 - lr), (w2 - lr, h2 - lr)):
                cv2.circle(lov, (cx, cy), lr, color, -1, cv2.LINE_AA)
            cv2.addWeighted(lov, badge_alpha, lroi, 1.0 - badge_alpha, 0, dst=lroi)
            cv2.putText(output_frame, str(corner_label), (lx0 + lpad, ly0 + lpad + lh),
                        font, scale, tuple(int(c) for c in new_ts_font_color),
                        thick, cv2.LINE_AA)
        return output_frame

    # 2. Hide the area with a filled rectangle (legacy solid-strip mode)
    cv2.rectangle(output_frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h),
                  hide_rect_color, -1)

    # 4. Determine font scale for the new timestamp
    if new_ts_font_scale is None:
        calculated_scale = rect_h * 0.022 # Factor for font scale based on rect height
        final_new_ts_font_scale = np.clip(calculated_scale, 0.3, 2.5) # Min 0.3, Max 2.5
    else:
        final_new_ts_font_scale = new_ts_font_scale
    
    final_new_ts_font_thickness = max(1, int(final_new_ts_font_scale + 0.5)) if new_ts_font_thickness == 1 else new_ts_font_thickness

    # 5. Determine position for the new timestamp
    if new_ts_position_on_rect:
        # Calculate padding in pixels. Using smaller dimension of rect for reference.
        # This makes padding more consistent if rect is very wide or very tall.
        padding_ref_dim = min(rect_w, rect_h)
        padding_pixels = int(padding_ref_dim * new_ts_padding_ratio)

        (text_w_px, text_h_above_baseline_px), baseline_px = cv2.getTextSize(
            time_string, new_ts_font, final_new_ts_font_scale, final_new_ts_font_thickness
        )
        full_text_render_height_px = text_h_above_baseline_px + baseline_px

        # Horizontal centering
        available_w_for_text = rect_w - (2 * padding_pixels)
        if text_w_px > available_w_for_text: # Text wider than available space
            ts_x_pos = rect_x + padding_pixels # Align to left padding
        else:
            horizontal_offset = (available_w_for_text - text_w_px) // 2
            ts_x_pos = rect_x + padding_pixels + horizontal_offset

        # Vertical centering
        available_h_for_text = rect_h - (2 * padding_pixels)
        if full_text_render_height_px > available_h_for_text: # Text taller than available space
            # Align top of text (baseline - text_h_above_baseline) with top padding
            ts_y_pos = rect_y + padding_pixels + text_h_above_baseline_px
        else:
            # Text fits, center it vertically
            vertical_offset = (available_h_for_text - full_text_render_height_px) // 2
            ts_y_pos = rect_y + padding_pixels + vertical_offset + text_h_above_baseline_px
        
        final_ts_position = (ts_x_pos, ts_y_pos)

    elif new_ts_custom_position:
        final_ts_position = new_ts_custom_position
    else:
        final_ts_position = (10, frame_h - 10) # Fallback

    # 6. Put the new timestamp on the frame
    cv2.putText(output_frame,
                time_string,
                final_ts_position,
                new_ts_font,
                final_new_ts_font_scale,
                new_ts_font_color,
                final_new_ts_font_thickness,
                cv2.LINE_AA)

    return output_frame

PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 165, 255), (128, 0, 128), (0, 255, 127), (128, 128, 0),
    (127, 255, 212), (255, 105, 180), (75, 0, 130), (255, 140, 0), (0, 128, 128)
]

def get_legend_layout(classes, class_to_label, text_scale, dot_size, padding=15, line_spacing=10):
    """Helper to calculate legend dimensions shared by all functions."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, int(text_scale * 2))
    max_text_w, total_text_h, rows = 0, 0, []
    
    unique_class_indices = sorted(list(set(classes)))

    for cls_idx in unique_class_indices:
        label = class_to_label.get(cls_idx, f"Class {cls_idx}")
        (t_w, t_h), _ = cv2.getTextSize(label, font, text_scale, text_thickness)
        max_text_w = max(max_text_w, t_w)
        total_text_h += t_h + line_spacing
        rows.append({'label': label, 'color': PALETTE[cls_idx % len(PALETTE)], 'h': t_h})

    legend_w = max_text_w + (dot_size * 2) + (padding * 3)
    legend_h = total_text_h + padding
    return legend_w, legend_h, rows, padding, line_spacing, font, text_thickness

def draw_common_elements(frame, boxes, classes, line_thickness, ids=None, text_scale=0.6):
    """Draws the boxes and optional IDs (common to all versions)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, int(text_scale * 2))
    
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = [int(c) for c in box]
        color = PALETTE[classes[i] % len(PALETTE)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, line_thickness)
        
        # Draw ID if provided and not None
        if ids is not None and i < len(ids) and ids[i] is not None:
            id_text = str(ids[i])
            (t_w, t_h), baseline = cv2.getTextSize(id_text, font, text_scale, text_thickness)
            
            padding = 2
            bg_w = t_w + padding * 2
            bg_h = t_h + padding * 2
            
            # Anchor at top-left. Prefer outside (above) but fallback to inside if it clips the frame
            if y_min - bg_h >= 0:
                bg_y1 = y_min - bg_h
                bg_y2 = y_min
                text_y = y_min - padding
            else:
                bg_y1 = y_min
                bg_y2 = y_min + bg_h
                text_y = y_min + t_h + padding
                
            bg_x1 = x_min
            bg_x2 = x_min + bg_w
            
            # Draw filled background box (same color as bounding box)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            
            # Calculate opposite color (255 - B, 255 - G, 255 - R)
            opp_color = (255 - color[0], 255 - color[1], 255 - color[2])
            
            # Draw ID text
            text_x = x_min + padding
            cv2.putText(frame, id_text, (text_x, text_y), font, text_scale, opp_color, text_thickness, cv2.LINE_AA)

def draw_boxes(frame, boxes, classes, class_to_label, confidences=None, # NOTE: for legacy support
                       line_thickness=2, legend_pos='top-right', legend_opacity=0.6, 
                       legend_corner_radius=15, text_scale=0.6, dot_size=6,
                       auto_scale=True, scale_reference_width=1920, ids=None):
    """
    Draws bounding boxes, optional IDs, and a glass-morphism legend on a frame.

    Args:
        frame (np.ndarray): Input image (BGR).
        boxes (list): List of [x_min, y_min, x_max, y_max] coordinates.
        classes (list): Class indices for each box.
        class_to_label (dict): Mapping from class index to label name.
        confidences (list, optional): Legacy parameter, not used.
        line_thickness (int): Box border thickness. Default: 2.
        legend_pos (str): Legend position ('top-left', 'top-right', 'bottom-left', 'bottom-right'). Default: 'top-right'.
        legend_opacity (float): Legend background opacity (0.0-1.0). Default: 0.6.
        legend_corner_radius (int): Legend corner radius in pixels. Default: 15.
        text_scale (float): Base text scale. Default: 0.6.
        dot_size (int): Base legend dot size. Default: 6.
        auto_scale (bool): Enable automatic scaling based on frame width. Default: True.
        scale_reference_width (int): Reference width for scaling (e.g., 1920 for Full HD). Default: 1920.
        ids (list, optional): List of IDs corresponding to each box. Use None for boxes without an ID. Default: None.

    Returns:
        np.ndarray: Frame with boxes, IDs, and legend drawn.
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Auto-scale legend elements based on frame width
    if auto_scale:
        scale_factor = frame_w / scale_reference_width
        text_scale = text_scale * scale_factor
        dot_size = int(dot_size * scale_factor)
        legend_corner_radius = int(legend_corner_radius * scale_factor)
        padding = int(15 * scale_factor)
        line_spacing = int(10 * scale_factor)
    else:
        padding = 15
        line_spacing = 10
    
    # Pass the ids and scaled text_scale to the drawing helper
    draw_common_elements(frame, boxes, classes, line_thickness, ids=ids, text_scale=text_scale)
    if not classes: return frame

    l_w, l_h, rows, pad, space, font, thick = get_legend_layout(classes, class_to_label, text_scale, dot_size, padding, line_spacing)
    
    margin = 20
    if legend_pos == 'top-left': lx, ly = margin, margin
    elif legend_pos == 'top-right': lx, ly = frame_w - l_w - margin, margin
    elif legend_pos == 'bottom-left': lx, ly = margin, frame_h - l_h - margin
    else: lx, ly = frame_w - l_w - margin, frame_h - l_h - margin
    lx, ly = max(0, min(lx, frame_w - l_w)), max(0, min(ly, frame_h - l_h))
    ex, ey = lx + l_w, ly + l_h

    # --- Downscale -> Blur -> Upscale ---
    roi = frame[ly:ey, lx:ex]
    
    # 1. Downscale by 4x (0.25)
    small_roi = cv2.resize(roi, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    
    # 2. Blur small image 
    blurred_small = cv2.GaussianBlur(small_roi, (5, 5), 0)
    
    # 3. Upscale back
    blurred_roi = cv2.resize(blurred_small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # 4. Integer blending
    glass_roi = cv2.addWeighted(blurred_roi, 1 - legend_opacity, np.full_like(roi, 255), legend_opacity, 0)
    
    # 5. Bitwise Masking
    mask = np.zeros((l_h, l_w), dtype=np.uint8)
    r = legend_corner_radius
    cv2.rectangle(mask, (r, 0), (l_w - r, l_h), 255, -1)
    cv2.rectangle(mask, (0, r), (l_w, l_h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (l_w - r, r), r, 255, -1)
    cv2.circle(mask, (r, l_h - r), r, 255, -1)
    cv2.circle(mask, (l_w - r, l_h - r), r, 255, -1)
    
    mask_inv = cv2.bitwise_not(mask)
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    roi_fg = cv2.bitwise_and(glass_roi, glass_roi, mask=mask)
    frame[ly:ey, lx:ex] = cv2.add(roi_bg, roi_fg)

    cy = ly + pad + rows[0]['h'] // 2
    for row in rows:
        cv2.circle(frame, (lx + pad + dot_size, cy - (row['h'] // 4)), dot_size, row['color'], -1)
        cv2.putText(frame, row['label'], (lx + pad * 2 + dot_size * 2, cy), font, text_scale, (0, 0, 0), thick, cv2.LINE_AA)
        cy += row['h'] + space
    return frame
