# BookScan App

A web application that identifies books from bookshelf photos using computer vision and OCR.

## Features

- Upload photos of bookshelves
- Automatically detect individual books
- Extract text from book spines
- View results with visual feedback
- Modern, responsive user interface

## Requirements

- Python 3.6+
- Tesseract OCR engine
- OpenCV
- Flask
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Make sure Tesseract is in your PATH or update the path in `src/book_detector.py`

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload a photo of your bookshelf
4. View the detected books and extracted text

## Tips for Best Results

- Take photos in good lighting conditions
- Ensure book spines are clearly visible
- Avoid glare on book covers
- Position camera parallel to the bookshelf
- Make sure text on book spines is readable

## Project Structure

- `app.py`: Main Flask application
- `src/book_detector.py`: Book detection and text extraction logic
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JS, uploaded images, results)

## License

MIT License
