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
    inplace=False
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

    # 2. Hide the area with a filled rectangle
    cv2.rectangle(output_frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h),
                  hide_rect_color, -1)

    # 3. Get current time and format it
    current_unix_time = time.time() if timestamp is None else timestamp
    dt_object = datetime.datetime.fromtimestamp(current_unix_time)
    time_string = dt_object.strftime("%Y-%m-%d %H:%M:%S")

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

def draw_common_elements(frame, boxes, classes, line_thickness):
    """Draws the boxes (common to all versions)."""
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = [int(c) for c in box]
        color = PALETTE[classes[i] % len(PALETTE)]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, line_thickness)

def draw_boxes(frame, boxes, classes, class_to_label, confidences=None, # NOTE: for legacy support
                       line_thickness=2, legend_pos='top-right', legend_opacity=0.6, 
                       legend_corner_radius=15, text_scale=0.6, dot_size=6,
                       auto_scale=True, scale_reference_width=1920):
    """
    Draws bounding boxes and a glass-morphism legend on a frame.

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

    Returns:
        np.ndarray: Frame with boxes and legend drawn.
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
    
    draw_common_elements(frame, boxes, classes, line_thickness)
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
    
    # 2. Blur small image (Tiny kernel is sufficient now)
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
