import logging
import cv2

def extract_barlines(gray, staff_lines):
    try:
        if gray is None or len(staff_lines) < 2:
            return []

        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        staff_lines = sorted(staff_lines)
        spacing = int(sum(
            staff_lines[i+1] - staff_lines[i] for i in range(len(staff_lines)-1)
        ) / (len(staff_lines)-1))

        top = max(0, staff_lines[0] - int(spacing * 0.7))
        bottom = min(gray.shape[0], staff_lines[-1] + int(spacing * 0.7))
        staff_region = gray[top:bottom]

        bin_img = cv2.adaptiveThreshold(
            staff_region, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            41, 7
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, int(spacing * 2))
        )
        vertical_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > spacing * 3.9:
                result.append((x, y + top, w, h))

        result.sort(key=lambda b: b[0])
        return result

    except Exception as ex:
        logging.getLogger(__name__).error(f"barline_extract_error: {ex}")
        return []
