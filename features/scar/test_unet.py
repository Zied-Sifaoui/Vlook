print("START")

import cv2
import torch
from unet_model import load_model

print("Model loading...")
model = load_model()
print("Model loaded")

img = cv2.imread("face.jpg")

if img is None:
    print("ERROR: Image not found")
    exit()

print("Image loaded")

img_resized = cv2.resize(img, (256, 256)) / 255.0

tensor = torch.tensor(img_resized).permute(2,0,1).unsqueeze(0).float()

print("Running model...")

with torch.no_grad():
    mask = model(tensor)

print("Model done")

mask = mask.squeeze().numpy()
mask = (mask > 0.5).astype("uint8") * 255

cv2.imshow("Original", img)
cv2.imshow("Mask", mask)

print("Showing windows...")

cv2.waitKey(0)
cv2.destroyAllWindows()