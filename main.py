import os
import re
import cv2
import easyocr
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import logging
import Levenshtein

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_plate(tokens):
    # Pisahkan ke huruf kapital
    tokens = [t.upper() for t in tokens]

    # Pisahkan huruf dan angka
    letters = [t for t in tokens if t.isalpha()]
    numbers = [t for t in tokens if t.isdigit()]

    # Ambil kode wilayah valid dari awal token
    kode_wilayah = next((t for t in tokens if t in kode_wilayah_valid), None)

    # Ambil angka 3-4 digit (kemungkinan nomor utama)
    nomor = max((t for t in numbers if 3 <= len(t) <= 4), key=int, default=None)

    # Ambil huruf akhiran yang bukan kode wilayah
    huruf_akhiran = [t for t in letters if t != kode_wilayah]
    kode_akhir = huruf_akhiran[0] if huruf_akhiran else None

    # Validasi keseluruhan
    if kode_wilayah and nomor and kode_akhir:
        return f'{kode_wilayah} {nomor} {kode_akhir}'

    # Jika sebagian, tampilkan hanya yang tersedia (tetap urut)
    partial = [t for t in [kode_wilayah, nomor, kode_akhir] if t]

    return ' '.join(partial)

def get_average_confidence(tokens, results):
    # Buat dictionary untuk pencocokan text → conf
    text_conf_map = {r['text'].upper(): r['conf'] for r in results}

    # Ambil confidence dari token yang digunakan
    conf_values = [text_conf_map.get(t.upper(), 0.0) for t in tokens if t.upper() in text_conf_map]

    if not conf_values:
        return 0.0

    return sum(conf_values) / len(conf_values)

def draw_text_dynamic(annotated_img, text, box, color=(255,255,255), bg_color=(255,0,0)):
    (x1, y1), (x2, y2) = tuple(box[0]), tuple(box[2])
    width = x2 - x1

    # Estimasi skala font berdasarkan panjang teks dan lebar box
    font_scale = max(min(width / (len(text) * 15), 2.0), 0.5)
    thickness = max(int(font_scale * 2), 2)

    # Background rectangle (untuk teks)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(annotated_img, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), bg_color, -1)

    # Tulisan OCR
    cv2.putText(annotated_img, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def cer(predicted: str, ground_truth: str) -> float:
    if not ground_truth:
        return 1.0 if predicted else 0.0
    return Levenshtein.distance(predicted, ground_truth) / len(ground_truth)

def extract_ground_truth_from_filename(filename: str) -> str:
    return os.path.splitext(filename)[0].upper().replace(" ", "")

# Load model YOLOv8
model = YOLO('./model/model black.pt')

# Inisialisasi EasyOCR
reader = easyocr.Reader(['id', 'en'])

# Input dan output directory
input_dir = './data1'
output_dir = "recognition_output"
cropped_dir = os.path.join(output_dir, "black cropped")
annotated_dir = os.path.join(output_dir, "black annotated")
thresholded_dir = os.path.join(output_dir, "black thresholded")
csv_path = os.path.join(output_dir, "black results.csv")

os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(thresholded_dir, exist_ok=True)
os.makedirs(annotated_dir, exist_ok=True)

kode_wilayah_valid = {
    'A', 'AA', 'AB', 'AD', 'AE', 'AG', 'B', 'BA', 'BB', 'BD', 'BE', 'BG',
    'BH', 'BK', 'BL', 'BM', 'BN', 'BP', 'D', 'DA', 'DB', 'DC',
    'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DP', 'DR', 'DT',
    'DW', 'E', 'EA', 'EB', 'ED', 'F', 'G', 'H', 'K', 'KB', 'KH', 'KT', 'KU',
    'L', 'M', 'N', 'P', 'PA', 'PB', 'R', 'S', 'T', 'W', 'Z'
}

results_data = []
image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
logger.info(f"Found {len(image_files)} images to process")

for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping unreadable file: {img_name}")
        continue

    annotated_img = img.copy()
    results = model(img)[0]

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Crop dan deskew
        plate_img = img[y1:y2, x1:x2]
        cropped_filename = f"{os.path.splitext(img_name)[0]}_plate_{i + 1}.jpg"
        cropped_path = os.path.join(cropped_dir, cropped_filename)
        cv2.imwrite(cropped_path, plate_img)

        # Grayscale + thresholding
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresholded_filename = f"{os.path.splitext(img_name)[0]}_plate_{i + 1}.jpg"
        thresholded_path = os.path.join(thresholded_dir, thresholded_filename)
        cv2.imwrite(thresholded_path, plate_thresh)

        # OCR
        ocr_result = reader.readtext(plate_thresh)
        filtered_texts = []

        # Simpan OCR untuk perhitungan confidence
        ocr_dicts = []
        for (_, text, prob) in ocr_result:
            if prob > 0.3:
                # Pisah berdasarkan spasi atau kombinasi huruf/angka
                tokens = re.findall(r'[A-Z]+|\d+', text.upper())
                filtered_texts.extend(tokens)
                # Pecah menjadi token individual
                for token in tokens:
                    ocr_dicts.append({'text': token, 'conf': prob})

        final_result = extract_plate(filtered_texts)
        avg_conf = get_average_confidence(final_result.split(), ocr_dicts)
        # Jika kosong atau confidence sangat rendah, beri label "No text detected"
        if not final_result or avg_conf == 0.0:
            final_result = "No text detected"
        elif not final_result or conf < 0.3:
            continue
        print("Token terfilter:", filtered_texts)
        '''if not final_result or conf < 0.3:
            continue'''
        # Ambil ground truth dari nama file
        ground_truth = extract_ground_truth_from_filename(img_name)

        # Hitung nilai CER
        cer_value = 1.0 if final_result == "No text detected" else cer(final_result.replace(" ", ""), ground_truth.replace(" ", ""))
        print(f'Result: {final_result} (avg confidence: {avg_conf:.2f}) (CER: {cer_value:.2f})')
        print(pd.DataFrame(ocr_result, columns=['bbox', 'text', 'conf']))

        # Koordinat untuk anotasi teks
        text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
        # Format teks hasil rekognisi
        text_label = f'{final_result} ({avg_conf:.2f}) CER: {cer_value:.2f}'

        # Gambar anotasi dinamis hanya untuk hasil yang valid
        '''if final_result:
            draw_text_dynamic(
                annotated_img,
                text_label,
                [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                color=(255, 255, 255),
                bg_color=(255, 0, 0)
            )'''

        # Gambar kotak ke annotated_img
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # Hitung ukuran teks
        (text_width, text_height), baseline = cv2.getTextSize(
            text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        # Tentukan posisi teks
        text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10
        # Gambar background rectangle
        cv2.rectangle(annotated_img,
                      (text_x, text_y - text_height - baseline),
                      (text_x + text_width, text_y + baseline),
                      (255, 0, 0), -1)
        # Gambar teks di atas background
        cv2.putText(annotated_img, text_label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Tambahkan data ke CSV
        results_data.append({
            "image": img_name,
            "bbox (x1, x2, y1, y2)": (x1, x2, y1, y2),
            "bbox confidence": round(conf, 2),
            "ocr confidence": round(avg_conf, 2),
            "plate text": final_result,
            "ground_truth": ground_truth,
            "CER": round(cer_value, 2),
            "cropped image": os.path.relpath(cropped_path, start=output_dir)
        })

        # Visualisasi (opsional)
        '''plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Plate")
        plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 2)
        plt.title("Grayscale")
        plt.imshow(plate_gray, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title("Thresholded")
        plt.imshow(plate_thresh, cmap='gray')
        plt.show()'''

    # Simpan gambar yang sudah di-annotate
    annotated_path = os.path.join(annotated_dir, img_name)
    cv2.imwrite(annotated_path, annotated_img)

    # Menampilkan gambar hasil anotasi
    '''plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Detections in {img_name}')
    plt.axis('off')
    plt.show()'''

# Simpan hasil ke CSV
df = pd.DataFrame(results_data)
df.to_csv(csv_path, index=False)
logger.info(f"✅ Total plates detected: {len(results_data)}")
logger.info(f"✅ CSV saved to: {csv_path}")
logger.info(f"📸 Cropped plates: {cropped_dir}")
logger.info(f"🖼️ Annotated images: {annotated_dir}")