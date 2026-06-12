import cv2
import numpy as np
import os as _os

_BASE = _os.path.dirname(_os.path.abspath(__file__))
_ASSETS = _os.path.join(_BASE, '..', 'assets', 'images')

def make_transparent(input_path, output_path):
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        print("Could not find image!")
        return

    # Convert to grayscale to find the background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask: everything that ISN'T the background
    # Adjust the '10' if there are still stray dark pixels
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Smooth the mask edges slightly so it's not jagged
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Split BGR channels and add the mask as the Alpha channel
    b, g, r = cv2.split(img)
    rgba = [b, g, r, mask]
    dst = cv2.merge(rgba, 4)

    # Save as PNG
    cv2.imwrite(output_path, dst)
    print(f"Success! {output_path} is now transparent.")

# Run it
make_transparent(_os.path.join(_ASSETS, "nose.png"))