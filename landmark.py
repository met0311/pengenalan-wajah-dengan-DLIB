import face_recognition
import cv2
import numpy as np

NAMA_GAMBAR = "test_mateus 1.jpeg" 

print(f"Memuat gambar: {NAMA_GAMBAR}...")
try:
    image_rgb = face_recognition.load_image_file(NAMA_GAMBAR)
except FileNotFoundError:
    print(f"Error: File '{NAMA_GAMBAR}' tidak ditemukan.")
    exit()
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
face_landmarks_list = face_recognition.face_landmarks(image_rgb)
print(f"Ditemukan {len(face_landmarks_list)} wajah dalam gambar.")

for face_landmarks in face_landmarks_list:
    for facial_feature in face_landmarks.keys():
        points = face_landmarks[facial_feature]
        for point in points:
            cv2.circle(image_bgr, point, 4, (0, 255, 0), -1)

WINDOW_TITLE = "Hasil Deteksi Facial Landmark" 
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

cv2.imshow(WINDOW_TITLE, image_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()