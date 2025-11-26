import logging
import cv2


def extract_barlines(gray, staff_lines):
    try:
        # normalize to grayscale
        if gray is None:
            return []

        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        if len(staff_lines) < 2:
            return []

        spacing = staff_lines[1] - staff_lines[0]

        top = max(0, staff_lines[0] - spacing)
        bottom = min(gray.shape[0], staff_lines[-1] + spacing)

        staff_region = gray[top:bottom, :]

        # binaryzacja
        _, bin_img = cv2.threshold(
            staff_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # kernel pionowy
        h = bin_img.shape[0]
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.6)))

        vertical_lines = cv2.morphologyEx(
            bin_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2
        )

        contours, _ = cv2.findContours(
            vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if h > spacing * 3:
                logging.getLogger(__name__).debug(
                    f"Detected barline at x={x}, y={y + top}, w={w}, h={h}"
                )
                result.append((x, y + top, w, h))

        result.sort(key=lambda b: b[0])
        return result[1:]  # skip first barline

    except Exception as ex:
        logging.getLogger(__name__).error("barline_extract_error:", ex)
        return []
