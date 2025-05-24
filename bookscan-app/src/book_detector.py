import cv2
import numpy as np
import pytesseract
import os

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

class BookDetector:
    def __init__(self, min_book_height=100, min_book_width=20):
        self.min_book_height = min_book_height
        self.min_book_width = min_book_width

    def detect_books(self, image_path):
        """
        Detect individual books in a bookshelf image and extract text from each spine.
        Displays intermediate images for debugging.

        Args:
            image_path: Path to the image file

        Returns:
            List of dicts containing 'bbox' and 'text', and the visualization path
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        img_vis = img.copy()

        # Debug: show original
        cv2.imshow("Original Image", img)
        cv2.waitKey(0); cv2.destroyWindow("Original Image")

        # 1) Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", gray)
        cv2.waitKey(0); cv2.destroyWindow("Gray")

        # Compute vertical gradient
        grad = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad = cv2.convertScaleAbs(grad)
        cv2.imshow("Vertical Gradient", grad)
        cv2.waitKey(0); cv2.destroyWindow("Vertical Gradient")

        # 2) Threshold gradient
        _, binarized = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
        cv2.imshow("Thresholded Gradient", binarized)
        cv2.waitKey(0); cv2.destroyWindow("Thresholded Gradient")

        # Close vertical gaps
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT,
                                             (3, self.min_book_height // 2))
        closed = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel_v, iterations=2)
        cv2.imshow("Closed Verticals", closed)
        cv2.waitKey(0); cv2.destroyWindow("Closed Verticals")

        # 3) Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h > self.min_book_height and w > self.min_book_width:
                raw_boxes.append((x, y, w, h))

        # 4) Merge overlapping boxes
        boxes = self._merge_overlapping_boxes(raw_boxes)

        # 5) Extract text from each ROI and draw boxes
        books = []
        for (x, y, w, h) in boxes:
            roi = img[y:y+h, x:x+w]
            text = self._extract_text_from_roi(roi)
            if text:
                books.append({'bbox': (x, y, w, h), 'text': text})
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 6) Save visualization
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(output_dir, f"{base}_detected.jpg")
        cv2.imwrite(vis_path, img_vis)

        return books, vis_path

    def process_image(self, image_path):
        """
        Wrapper around detect_books to return formatted results and visualization.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple (results, visualization_path)
        """
        books, vis_path = self.detect_books(image_path)
        results = []
        for i, b in enumerate(books, start=1):
            results.append({
                'id': i,
                'text': b['text'],
                'position': b['bbox']
            })
        return results, vis_path

    @staticmethod
    def _merge_overlapping_boxes(boxes):
        """
        Merge boxes whose x-ranges overlap significantly.
        """
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[0])
        merged = [boxes[0]]
        for (x1, y1, w1, h1) in boxes[1:]:
            x0, y0, w0, h0 = merged[-1]
            if x1 < x0 + w0 * 0.5:
                nx = min(x0, x1)
                ny = min(y0, y1)
                nw = max(x0 + w0, x1 + w1) - nx
                nh = max(y0 + h0, y1 + h1) - ny
                merged[-1] = (nx, ny, nw, nh)
            else:
                merged.append((x1, y1, w1, h1))
        return merged

    def _extract_text_from_roi(self, roi):
        """
        Preprocess a spine ROI and extract text via Tesseract.
        Displays intermediate images for debugging.
        """
        # Resize for better OCR
        roi_resized = cv2.resize(roi, (0, 0), fx=2, fy=2)
        cv2.imshow("ROI Resized", roi_resized)
        cv2.waitKey(0); cv2.destroyWindow("ROI Resized")

        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        cv2.imshow("ROI Gray", gray)
        cv2.waitKey(0); cv2.destroyWindow("ROI Gray")

        # Rotate if vertical
        h, w = gray.shape
        # if h > w:
        #     gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        #     cv2.imshow("ROI Rotated", gray)
        #     cv2.waitKey(0); cv2.destroyWindow("ROI Rotated")

        # Boost contrast
        gray_eq = cv2.equalizeHist(gray)
        cv2.imshow("ROI Equalized", gray_eq)
        cv2.waitKey(0); cv2.destroyWindow("ROI Equalized")

        # Binarize (OTSU)
        _, thresh = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("ROI Thresholded", thresh)
        cv2.waitKey(0); cv2.destroyWindow("ROI Thresholded")

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
        cv2.imshow("ROI Denoised", denoised)
        cv2.waitKey(0); cv2.destroyWindow("ROI Denoised")

        # OCR as single block
        config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=' \
                 r'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        text = pytesseract.image_to_string(denoised, config=config)

        # Cleanup
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return ' '.join(lines) if lines else ''


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect books and extract text from spines.'
    )
    parser.add_argument('image', help='Path to the input image')
    args = parser.parse_args()

    detector = BookDetector()
    results, vis_path = detector.process_image(args.image)

    print(f'Found {len(results)} books:')
    for b in results:
        print(f"{b['id']}: '{b['text']}' at {b['position']}")
    print(f'Visualization saved to: {vis_path}')