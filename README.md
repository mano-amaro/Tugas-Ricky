# Tugas-Ricky
Face Recognitions

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import shutil # Ditambahkan untuk kompresi file
from google.colab import files

# --- Konfigurasi File ---
HAAR_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
HAAR_CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'

# Folder Colab tempat file output akan disimpan sementara
# Jika Anda menjalankan kode ini di Python lokal (bukan Colab), Anda bisa mengubahnya menjadi:
# OUTPUT_FOLDER = 'C:/Users/ricky/Downloads/Image'
OUTPUT_FOLDER = 'image_data_output' 
LABEL_FILENAME = 'labels.txt'
ZIP_FILENAME = 'face_data_collection.zip'

# --- Langkah 1: Mengunduh Haar Cascade ---
print("🚀 Mengunduh dan menginisialisasi file Haar Cascade...")
!wget -q {HAAR_CASCADE_URL}
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILENAME)
print("✅ Haar Cascade berhasil diinisialisasi.")
print("-" * 30)

# --- Langkah 2: Mempersiapkan Folder Output ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"📁 Folder output Colab '{OUTPUT_FOLDER}' berhasil dibuat.")
else:
    print(f"📁 Folder output Colab '{OUTPUT_FOLDER}' sudah ada.")
# Menghapus file labels.txt lama jika ada, agar data tidak menumpuk
if os.path.exists(LABEL_FILENAME):
    os.remove(LABEL_FILENAME)
print("-" * 30)

# --- Langkah 3: Mengunggah Gambar & Meminta Label ---
image_name = None
print("Silakan unggah gambar Anda (Contoh: BaHlil.jpg).")

try:
    uploaded = files.upload()
    if uploaded:
        image_name = list(uploaded.keys())[0]
        print(f"🖼 Gambar yang diunggah: {image_name}")
    else:
        print("❌ Tidak ada gambar diunggah. Kode berhenti.")
        exit()
except Exception as e:
    print(f"❌ Error saat mengunggah: {e}. Kode berhenti.")
    exit()

person_label = input("🏷 Masukkan Label untuk wajah ini (misalnya: 'dilan' atau ID): ").strip()
if not person_label:
    person_label = "unknown"
    print("Label kosong, menggunakan 'unknown'.")
print("-" * 30)


# --- Langkah 4: Fungsi Deteksi Wajah, Cropping, dan Penyimpanan Data ---
def process_faces_for_recognition(image_path, cascade_classifier, output_dir, label):
    """
    Melakukan deteksi wajah, cropping, resize ke 50x50, menyimpan gambar dan label.
    """
    TARGET_SIZE = (50, 50) 

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"ERROR: Tidak dapat memuat gambar dari {image_path}")
        return 0, []

    img_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"Ditemukan {len(faces)} wajah pada gambar.")
    
    saved_files = []

    # Membuka file label dalam mode append ('a') di root Colab
    with open(LABEL_FILENAME, 'a') as f_label: 
        for i, (x, y, w, h) in enumerate(faces):
            face_crop = img_bgr[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            
            timestamp = int(time.time() * 1000)
            # Path output di dalam folder Colab
            output_filename = os.path.join(output_dir, f"{label}{timestamp}{i}.jpg")
            
            # Menyimpan gambar
            cv2.imwrite(output_filename, face_resized)
            
            # Menyimpan label ke file teks. Path harus relatif atau absolut yang konsisten.
            # Kita gunakan path Colab yang telah kita definisikan.
            f_label.write(f"{output_filename},{label}\n")
            
            saved_files.append(output_filename)
            print(f"   - Wajah {i+1} tersimpan sebagai: {output_filename}")
            
            cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_display, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_display)
    plt.title(f"Deteksi Wajah & Data Prep | Label: {label}")
    plt.axis('off')
    plt.show()
    
    return len(faces), saved_files

# --- Langkah 5: Eksekusi, Kompresi, dan Unduh ---
if os.path.exists(image_name):
    print("⚙ Memproses gambar...")
    face_count, files_saved = process_faces_for_recognition(image_name, face_cascade, OUTPUT_FOLDER, person_label)
    print("-" * 30)
    print(f"✅ Selesai! Total {face_count} wajah diproses dan disimpan di {OUTPUT_FOLDER}.")
    
    # Pindahkan labels.txt ke folder output agar ikut ter-zip
    if os.path.exists(LABEL_FILENAME):
        shutil.move(LABEL_FILENAME, os.path.join(OUTPUT_FOLDER, LABEL_FILENAME))
        
    # Kompres folder output menjadi ZIP (agar mudah diunduh)
    print(f"📦 Mengompres hasil ke {ZIP_FILENAME}...")
    shutil.make_archive(OUTPUT_FOLDER, 'zip', OUTPUT_FOLDER)
    
    # Unduh file ZIP (agar Anda bisa memindahkannya ke C:\Users\ricky\Downloads\Image)
    print(f"⬇ Mengunduh {ZIP_FILENAME}...")
    files.download(f'{OUTPUT_FOLDER}.zip')
    
    print("\n➡ *Lanjut ke recognition*")
else:
    print("❌ ERROR FATAL: File gambar tidak ditemukan.")
