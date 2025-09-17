import cv2
from skimage import feature
from skimage import exposure
import numpy as np

NAMA_GAMBAR = "test_mateus 1.jpeg" 

print(f"Memuat gambar: {NAMA_GAMBAR}...")
image = cv2.imread(NAMA_GAMBAR)
if image is None:
    print(f"Error: File '{NAMA_GAMBAR}' tidak ditemukan atau format tidak didukung.")
    exit()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(H, hog_image) = feature.hog(gray_image, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys",
                            visualize=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 255))

hog_image_rescaled = hog_image_rescaled.astype("uint8")

cv2.namedWindow("Gambar Hitam-Putih (Input)", cv2.WINDOW_NORMAL)
cv2.imshow("Gambar Hitam-Putih (Input)", gray_image)

cv2.namedWindow("Visualisasi HOG (Output)", cv2.WINDOW_NORMAL)
cv2.imshow("Visualisasi HOG (Output)", hog_image_rescaled)

print("\nDua jendela visualisasi telah muncul.")
print("Tekan tombol apa saja di keyboard untuk menutup.")

cv2.waitKey(0)
cv2.destroyAllWindows()