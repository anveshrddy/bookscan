import cv2
import numpy as np
import pytesseract
import os

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

class BookDetector:
    def __init__(self):
        self.min_book_height = 100  # Minimum height for a book in pixels
        self.min_book_width = 20    # Minimum width for a book in pixels

    def detect_books(self, image_path):
        """
        Detect individual books in a bookshelf image and extract text from each spine
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing book information (bounding box, extracted text)
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Create a copy for visualization
        img_vis = img.copy()

        # Rotate the image if necessary
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate the edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and shape to identify potential books
        books = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size
            if h > self.min_book_height and w > self.min_book_width:
                # Calculate aspect ratio
                aspect_ratio = h / w
                
                # Most books are taller than they are wide
                if aspect_ratio > 1.2:
                    # Extract the region of interest (ROI)
                    roi = img[y:y+h, x:x+w]
                    
                    # Extract text from the ROI
                    text = self._extract_text_from_roi(roi)
                    
                    # Add to books list if text was found
                    if text:
                        books.append({
                            'bbox': (x, y, w, h),
                            'text': text,
                            'roi': roi
                        })
                    
                    # Draw rectangle on visualization image
                    cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save visualization image
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filename without extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_filename}_detected.jpg")
        cv2.imwrite(output_path, img_vis)
        
        return books, output_path
    
    def _extract_text_from_roi(self, roi):
        """
        Extract text from a region of interest (book spine)
        
        Args:
            roi: Region of interest (book spine image)
            
        Returns:
            Extracted text
        """
        # Resize for better OCR performance
        roi = cv2.resize(roi, (0, 0), fx=2, fy=2)
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
        
        # OCR configuration
        config = r'--oem 1 --psm 12'
        
        # Extract text
        text = pytesseract.image_to_string(denoised, config=config)
        
        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return ' '.join(lines) if lines else ""
        
    def process_image(self, image_path):
        """
        Process an image and return detected books
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of books with their extracted text
        """
        books, vis_path = self.detect_books(image_path)
        
        # Format results
        results = []
        for i, book in enumerate(books):
            results.append({
                'id': i + 1,
                'text': book['text'],
                'position': book['bbox']
            })
            
        return results, vis_path


if __name__ == "__main__":
    # Test the detector
    detector = BookDetector()
    image_path = "D:/Side_Projects/ComputerVision/bookscan-app/images/sample-image.jpeg"
    books, vis_path = detector.process_image(image_path)
    
    print(f"Found {len(books)} books:")
    for book in books:
        print(f"Book {book['id']}: {book['text']}")
    print(f"Visualization saved to: {vis_path}")
