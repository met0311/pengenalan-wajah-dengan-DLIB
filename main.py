import face_recognition
import cv2
import os
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
import time
import telegram
import asyncio
import pickle
import requests

TELEGRAM_TOKEN = '8065148553:AAEUcFM5nGLw2CQ5AXlkDGzeS5lZjQxXHoA'
TELEGRAM_CHAT_ID = '970886930'
ESP8266_IP = "192.168.220.223"
KNOWN_FACES_DIR = 'known_faces'
TEST_DATASET_DIR = 'test_dataset'
KNOWN_ENCODINGS_FILE = 'known_face_encodings.pkl'
TOLERANCE = 0.45


def load_known_faces(known_faces_dir, encodings_file, force_retrain=False):
    
    # Training Dataset
    if not force_retrain and os.path.exists(encodings_file):
        print(f"Mencoba memuat data wajah dari {encodings_file}...")
        try:
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Berhasil memuat {len(data['encodings'])} data wajah.")
            return data['encodings'], data['names']
        except Exception as e:
            print(f"Gagal memuat file: {e}. Melakukan training ulang.")

    # Proses training dari direktori
    print(f"Memulai proses encoding wajah dari direktori: {known_faces_dir}...")
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(known_faces_dir):
        print(f"ERROR: Direktori '{known_faces_dir}' tidak ditemukan.")
        return [], []

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            try:
                print(f"-> Memproses {name}/{filename}...")
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"   [OK] Wajah '{name}' ditemukan dan di-encode.")
                    print(f"   Vektor 128-d: {encodings[0]}")
                else:
                    print(f"   [Warning] Tidak ada wajah terdeteksi di {filename}.")
            except Exception as e:
                print(f"   [ERROR] Gagal memproses {image_path}: {e}")

    if known_face_encodings:
        print(f"\nMenyimpan {len(known_face_encodings)} data wajah ke {encodings_file}.")
        with open(encodings_file, 'wb') as f:
            pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, f)
        print("Penyimpanan berhasil.")
    
    return known_face_encodings, known_face_names


async def send_telegram_notif(bot, chat_id, message):
    """ Telegram Notifikasi"""
    if not bot:
        return
    try:
        await bot.send_message(chat_id=chat_id, text=message)
        print(f"Notifikasi Telegram terkirim: '{message}'")
    except Exception as e:
        print(f"Gagal mengirim notifikasi Telegram: {e}")


async def trigger_esp8266():
    """Kirim sinyal HTTP GET ke ESP8266."""

    trigger_url = f"http://{ESP8266_IP}/trigger"
    print(f"Mengirim trigger ke ESP8266 di {trigger_url}...")

    def send_request():
        try:
            response = requests.get(trigger_url, timeout=5)
            if response.status_code == 200:
                print("Berhasil mentrigger ESP8266.")
            else:
                print(f"Gagal mentrigger ESP8266, status: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error saat menghubungi ESP8266: {e}")

    
    await asyncio.to_thread(send_request)


def process_frame(frame_rgb, known_encodings, known_names):
    """Proses satu frame: deteksi, kenali, dan gambar kotak."""
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    recognized_names_in_frame = set()

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        print(f"-> Vektor 128-d dari wajah terdeteksi: {encoding}")
        name = "Tidak Dikenal"
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    distance = face_distances[best_match_index]
                    print(f"Wajah Dikenali: {name} (Nilai Euclidean: {distance:.2f})")
        
        recognized_names_in_frame.add(name)
        
        box_color = (0, 255, 0) if name != "Tidak Dikenal" else (255, 0, 0)
        draw.rectangle(((left, top), (right, bottom)), outline=box_color, width=3)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((left, bottom), name, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle(((left, bottom), (right, bottom + text_height + 10)), fill=box_color, outline=box_color)
        draw.text((left + 6, bottom + 5), name, fill=(255, 255, 255), font=font)
    
    return pil_image, list(recognized_names_in_frame)


async def start_live_detection(known_encodings, known_names, bot, chat_id):
    """Buka kamera dan mulai deteksi wajah secara real-time."""
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    print("\n--- Memulai Deteksi Live (Tekan 'q' untuk keluar) ---")
    
    notified_persons = {}
    last_unknown_notif = 0
    COOLDOWN_SECONDS = 10

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_pil_image, recognized_names = process_frame(frame_rgb, known_encodings, known_names)
        processed_frame_bgr = cv2.cvtColor(np.array(processed_pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow('Deteksi Wajah', processed_frame_bgr)

        current_time = time.time()
        
        for name in recognized_names:
            if name != "Tidak Dikenal":
                if current_time - notified_persons.get(name, 0) > COOLDOWN_SECONDS:
                    message = f"Pintu Terbuka: '{name}' terdeteksi."
                    await send_telegram_notif(bot, chat_id, message)
                    await trigger_esp8266()
                    notified_persons[name] = current_time
            else:
                if current_time - last_unknown_notif > COOLDOWN_SECONDS:
                    await send_telegram_notif(bot, chat_id, "AWASS ORANG TIDAK DIKENAL")
                    last_unknown_notif = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Deteksi live dihentikan.")


def evaluate_model(test_dir, known_encodings, known_names):
    """Evaluasi akurasi model menggunakan gambar di folder test_dataset."""
    print(f"\n Memulai Evaluasi Model dari '{test_dir}' ")
    if not os.path.exists(test_dir):
        print(f"Error: Direktori uji '{test_dir}' tidak ditemukan.")
        return

    tp, fp, fn, tn = 0, 0, 0, 0 

    for ground_truth_name in os.listdir(test_dir):
        person_dir = os.path.join(test_dir, ground_truth_name)
        if not os.path.isdir(person_dir):
            continue
        
        print(f"\nMengevaluasi: '{ground_truth_name}'")
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            try:
                test_image = face_recognition.load_image_file(image_path)
                locations = face_recognition.face_locations(test_image)
                
                if not locations:
                    if ground_truth_name != 'tidak_dikenal':
                        fn += 1 # Seharusnya ada wajah, tapi tidak terdeteksi
                    continue

                test_encoding = face_recognition.face_encodings(test_image, locations)[0]
                matches = face_recognition.compare_faces(known_encodings, test_encoding, tolerance=TOLERANCE)
                
                predicted_name = "Tidak Dikenal"
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_encodings, test_encoding))
                    if matches[best_match_index]:
                        predicted_name = known_names[best_match_index]

                print(f"   - {filename}: [Seharusnya: {ground_truth_name}] -> [Prediksi: {predicted_name}]")
                
                is_known_gt = (ground_truth_name != 'tidak_dikenal')
                is_known_pred = (predicted_name != 'Tidak Dikenal')

                if is_known_gt and predicted_name == ground_truth_name:
                    tp += 1
                elif not is_known_gt and not is_known_pred: 
                    tn += 1
                elif not is_known_gt and is_known_pred:
                    fp += 1
                elif is_known_gt and not is_known_pred:
                    fn += 1
                elif is_known_gt and is_known_pred and predicted_name != ground_truth_name:
                    fp += 1 
                    fn += 1

            except Exception as e:
                print(f"   - Error saat memproses {filename}: {e}")

    print("\n--- Hasil Evaluasi ---")
    print(f"True Positives (Benar prediksi orang): {tp}")
    print(f"True Negatives (Benar prediksi tidak dikenal): {tn}")
    print(f"False Positives (Salah prediksi orang): {fp}")
    print(f"False Negatives (Gagal kenali orang): {fn}")
    
    try:
        precision = tp / (tp + fp)
        print(f"Presisi: {precision:.2%}")
    except ZeroDivisionError:
        print("Presisi: N/A (Tidak ada prediksi positif)")

    try:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.2%}")
    except ZeroDivisionError:
        print("Recall: N/A (Tidak ada TP atau FN)")

    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"Akurasi: {accuracy:.2%}")
    except ZeroDivisionError:
        print("Akurasi: N/A (Tidak ada data uji)")


async def main():

    parser = argparse.ArgumentParser(description="Deteksi wajah dengan Notifikasi Telegram.")
    parser.add_argument("--live", action="store_true", help="live dari kamera.")
    parser.add_argument("--evaluate", action="store_true", help="evaluasi")
    parser.add_argument("--retrain", action="store_true", help="training ulang")
    args = parser.parse_args()
    bot = telegram.Bot(token=TELEGRAM_TOKEN)

    # Muat atau training data wajah
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR, KNOWN_ENCODINGS_FILE, args.retrain)
    if not known_encodings:
        print("\nPERINGATAN: Tidak ada data wajah yang dimuat. Pengenalan tidak akan bekerja.")
        if not args.retrain:
             print("Coba jalankan dengan flag --retrain untuk membuat data wajah.")

    # Jalankan mode sesuai argumen
    if args.evaluate:
        evaluate_model(TEST_DATASET_DIR, known_encodings, known_names)
    elif args.live:
        await start_live_detection(known_encodings, known_names, bot, TELEGRAM_CHAT_ID)
    else:
        print("\nTidak ada mode yang dipilih. Gunakan --live atau --evaluate.")
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())