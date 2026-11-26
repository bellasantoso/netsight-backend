from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import os

# ============================================================
# FIREBASE ADMIN SDK
# ============================================================
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# ============================================================
# INISIALISASI FLASK
# ============================================================
app = Flask(__name__)

# ============================================================
# 1. LOAD MODEL, SCALER, DAN FIREBASE
# ============================================================
try:
    # --------------------------------------------------------
    # LOAD MODEL DNN
    # --------------------------------------------------------
    model = tf.keras.models.load_model('model_kualitas_jaringan.keras')

    # --------------------------------------------------------
    # LOAD SCALER
    # --------------------------------------------------------
    scaler = joblib.load('scaler_jaringan.pkl')
    print("Model dan Scaler berhasil dimuat ke dalam server!")

    # --------------------------------------------------------
    # INISIALISASI FIREBASE
    # --------------------------------------------------------
    if not firebase_admin._apps:
        key_path = os.environ.get("FIREBASE_KEY", "serviceAccountKey.json")
        cred = credentials.Certificate(key_path)

        firebase_admin.initialize_app(
            cred,
            {
                'databaseURL': 'https://netsight-fd473-default-rtdb.asia-southeast1.firebasedatabase.app'
            }
        )
        print("Firebase berhasil terhubung!")

except Exception as e:
    print(f"Error memuat model/scaler/Firebase: {e}")


# ============================================================
# 2. URUTAN KELAS DNN
# ============================================================
classes = ['Excellent', 'Good', 'Marginal', 'Poor']


# ============================================================
# 3. RULE-BASED ANALYSIS
# ============================================================
def analisis_rule_based(rsrp, rsrq, sinr):
    analisis = []
    indikasi = []
    rekomendasi = []
    prioritas = "Rendah"

    # ANALISIS RSRP
    if rsrp < -100:
        analisis.append("RSRP: Buruk")
        indikasi.append("Coverage lemah")
        rekomendasi.append("Optimasi electrical/mechanical tilt antena")
        rekomendasi.append("Evaluasi azimuth sektor terdekat")
        prioritas = "Tinggi"
    elif rsrp < -90:
        analisis.append("RSRP: Marginal")
        indikasi.append("Level sinyal relatif rendah")
        rekomendasi.append("Evaluasi coverage footprint sektor")
        prioritas = "Sedang"
    else:
        analisis.append("RSRP: Baik")

    # ANALISIS RSRQ
    if rsrq < -15:
        analisis.append("RSRQ: Buruk")
        indikasi.append("Kualitas radio rendah")
        rekomendasi.append("Evaluasi inter-cell interference")
        rekomendasi.append("Evaluasi overlap coverage antar sektor")
        prioritas = "Tinggi"
    elif rsrq < -10:
        analisis.append("RSRQ: Marginal")
        indikasi.append("Kualitas radio menurun")
        rekomendasi.append("Evaluasi konfigurasi radio pada sektor terkait")
    else:
        analisis.append("RSRQ: Baik")

    # ANALISIS SINR
    if sinr < 0:
        analisis.append("SINR: Buruk")
        indikasi.append("Potensi interferensi tinggi")
        rekomendasi.append("Evaluasi PCI dan neighbor cell")
        rekomendasi.append("Evaluasi frequency allocation / retuning")
        prioritas = "Tinggi"
    elif sinr < 10:
        analisis.append("SINR: Marginal")
        indikasi.append("Kualitas sinyal kurang optimal")
        rekomendasi.append("Evaluasi interferensi pada sel sekitar")
    else:
        analisis.append("SINR: Baik")

    # KONDISI NORMAL
    if len(indikasi) == 0:
        indikasi.append("Tidak ditemukan indikasi gangguan signifikan")
        rekomendasi.append("Tidak diperlukan tindakan optimasi")

    # HILANGKAN DATA DUPLIKAT
    indikasi = list(dict.fromkeys(indikasi))
    rekomendasi = list(dict.fromkeys(rekomendasi))

    return {
        "analisis": analisis,
        "indikasi": indikasi,
        "prioritas": prioritas,
        "rekomendasi": rekomendasi
    }


# ============================================================
# 4. ENDPOINT PREDIKSI
# ============================================================
@app.route('/predict_kualitas', methods=['POST'])
def predict_kualitas():
    try:
        data = request.get_json()
        
        # --- LOG UNTUK MELIHAT DATA DARI ANDROID ---
        print(f"\n[INFO] Data yang diterima dari HP: {data}")

        if data is None:
            return jsonify({"status": "error", "pesan": "Data JSON tidak ditemukan."}), 400

        # EKSTRAKSI KPI (TERMASUK NOMOR HP)
        rsrp = float(data['rsrp'])
        rsrq = float(data['rsrq'])
        sinr = float(data['sinr'])
        latitude = float(data.get('latitude', 0.0))
        longitude = float(data.get('longitude', 0.0))
        waktu = data.get('waktu', '')
        
        # MENANGKAP NOMOR HP DARI ANDROID
        phone = data.get('phone', 'unknown')

        # PREDIKSI DNN
        input_data = np.array([[rsrp, rsrq, sinr]])
        input_scaled = scaler.transform(input_data)
        prediksi_probs = model.predict(input_scaled, verbose=0)
        
        prediksi_index = np.argmax(prediksi_probs, axis=1)[0]
        kualitas_jaringan = classes[prediksi_index]
        confidence = float(np.max(prediksi_probs) * 100)

        # RULE-BASED ANALYSIS
        hasil_rule = analisis_rule_based(rsrp, rsrq, sinr)

        # PENYIAPAN DATA FIREBASE
        firebase_data = {
            "status": "success",
            "kualitas_jaringan": kualitas_jaringan,
            "kategori_ai": kualitas_jaringan,
            "confidence": f"{confidence:.2f}%",
            "latitude": latitude,
            "longitude": longitude,
            "rsrp": rsrp,
            "rsrq": rsrq,
            "sinr": sinr,
            "analisis": hasil_rule["analisis"],
            "indikasi": hasil_rule["indikasi"],
            "prioritas": hasil_rule["prioritas"],
            "rekomendasi": hasil_rule["rekomendasi"],
            "pesan": "Klasifikasi dan analisis jaringan berhasil."
        }

        if waktu:
            firebase_data["waktu"] = waktu

        # ==========================================
        # SIMPAN KE FIREBASE DI BAWAH FOLDER NOMOR HP
        # ==========================================
        if latitude != 0.0 and longitude != 0.0:
            ref = db.reference('RiwayatJaringan').child(phone)
            ref.push(firebase_data)
            print(f"Data lengkap! Berhasil disimpan ke Firebase di node: {phone}")
        else:
            print("Data kembar terdeteksi (lokasi 0.0). Tidak disave ke Firebase.")

        # RESPONSE KE ANDROID
        return jsonify(firebase_data), 200

    except Exception as e:
        print(f"Error endpoint: {e}")
        return jsonify({"status": "error", "pesan": str(e)}), 400


# ============================================================
# 5. ROUTE DEFAULT
# ============================================================
@app.route('/', methods=['GET'])
def home():
    return "Server Backend Pemantauan Kualitas Jaringan Aktif!"


# ============================================================
# 6. MENJALANKAN SERVER
# ============================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
