import cv2
import pytesseract

# 1. Point pytesseract at your Tesseract binary:
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def extract_text_from_book_spine(image_path):
    img = cv2.imread(image_path)
    
    # Resize for better OCR performance
    img = cv2.resize(img, (0, 0), fx=2, fy=2)  # Increase image resolution

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to improve binarization
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Rotate image if necessary
    (h, w) = thresh.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, 1.0)  # Adjust the angle if needed
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Denoise (Optional)
    denoised = cv2.fastNlMeansDenoising(rotated, None, 30, 7, 21)

    # 2. OCR with tuned PSM/OEM
    config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    raw = pytesseract.image_to_string(denoised, config=config)

    return [line.strip() for line in raw.split('\n') if line.strip()]

if __name__ == '__main__':
    spines = extract_text_from_book_spine("D:/Side_Projects/ComputerVision/bookscan-app/images/sample-image.jpeg")
    print(spines)
