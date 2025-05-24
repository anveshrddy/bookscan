#!/usr/bin/env python3
import json
import argparse
import os

from book_detector import BookDetector

def main():
    parser = argparse.ArgumentParser(
        description="Detect books in an image and extract spine text."
    )
    parser.add_argument(
        "image_path", help="Path to the bookshelf image file"
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Print JSON with indentation"
    )
    args = parser.parse_args()

    detector = BookDetector()
    try:
        results, vis_path = detector.process_image(args.image_path)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    output = {
        "books_found": len(results),
        "results": results,
        "visualization": vis_path
    }

    if args.pretty:
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps(output))

if __name__ == "__main__":
    main()
