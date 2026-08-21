from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

# 1. Load Model dan Scaler saat server mulai berjalan
try:
    model = tf.keras.models.load_model('model_kualitas_jaringan.keras')
    scaler = joblib.load('scaler_jaringan.pkl')
    print("Model dan Scaler berhasil dimuat ke dalam server!")
except Exception as e:
    print(f"Error memuat model/scaler: {e}")

# 2. Definisikan urutan kelas (Sesuai dengan urutan LabelEncoder saat training)
classes = ['Excellent', 'Good', 'Marginal', 'Poor']

# 3. Endpoint API untuk menerima data dari Android
@app.route('/predict_kualitas', methods=['POST'])
def predict_kualitas():
    try:
        # Menangkap data JSON yang dikirim oleh aplikasi Android
        data = request.get_json()
        
        # Mengekstrak parameter KPI
        rsrp = float(data['rsrp'])
        rsrq = float(data['rsrq'])
        sinr = float(data['sinr'])
        
        # Data lokasi (Disiapkan untuk fitur simpan ke Firebase nantinya)
        latitude = data.get('latitude', 0.0)
        longitude = data.get('longitude', 0.0)

        # Menyusun data ke dalam format array untuk diproses
        input_data = np.array([[rsrp, rsrq, sinr]])
        
        # PROSES INTI: Normalisasi data baru menggunakan scaler
        input_scaled = scaler.transform(input_data)
        
        # Prediksi menggunakan model DNN
        prediksi_probs = model.predict(input_scaled)
        prediksi_index = np.argmax(prediksi_probs, axis=1)[0]
        
        # Mendapatkan label teks
        kualitas_jaringan = classes[prediksi_index]
        confidence = float(np.max(prediksi_probs) * 100)

        # Format balasan ke Android
        response = {
            "status": "success",
            "kualitas_jaringan": kualitas_jaringan,
            "confidence": f"{confidence:.2f}%",
            "pesan": "Klasifikasi berhasil."
            # Di sini nanti kita bisa tambahkan logika insert ke Firebase
        }
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"status": "error", "pesan": str(e)}), 400

# Route default untuk memastikan server hidup
@app.route('/', methods=['GET'])
def home():
    return "Server Backend Pemantauan Kualitas Jaringan Aktif!"

if __name__ == '__main__':
    # Jalankan server. host='0.0.0.0' agar bisa diakses dari emulator/HP Android di jaringan yang sama
    app.run(host='0.0.0.0', port=5000, debug=True)