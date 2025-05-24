import os
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from book_detector import BookDetector

@pytest.fixture
def blank_image(tmp_path):
    # create a blank (black) image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    file = tmp_path / "blank.jpg"
    cv2.imwrite(str(file), img)
    return str(file)

def test_invalid_path_raises():
    bd = BookDetector()
    # path doesn’t exist → ValueError
    with pytest.raises(ValueError):
        bd.detect_books("this_file_does_not_exist.png")

def test_detect_books_on_blank(blank_image):
    bd = BookDetector()
    books, vis_path = bd.detect_books(blank_image)

    # No books should be found
    assert isinstance(books, list)
    assert books == []

    # Visualization file should still be written
    assert os.path.isfile(vis_path)

def test_process_image_wrapper(blank_image):
    bd = BookDetector()
    results, vis_path = bd.process_image(blank_image)

    # results is list of dicts, but blank → empty list
    assert isinstance(results, list)
    assert results == []

    # visualization exists
    assert os.path.exists(vis_path)
