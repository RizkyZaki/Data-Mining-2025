
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

# 1. Load Model yang sudah dilatih sebelumnya
model = joblib.load("D:/School/S4/DATMIN/PRAC/Data-Mining-2025/MODUL3/stroke_prediction_model.pkl")

# Judul
st.set_page_config(
    page_title="Dashboard Prediksi Risiko Stroke",
    page_icon="🧠",
)
st.title(" TP Modul 3 - Stroke Prediction App")
st.write("Masukkan data pasien untuk memprediksi kemungkinan terkena stroke.")

st.markdown("---")
st.subheader("Evaluasi Model")

# 2. Load dan persiapkan data sama seperti saat melatih model sebelumnya
df = pd.read_csv('D:/School/S4/DATMIN/PRAC/Data-Mining-2025/MODUL3/dataset_StrokePredicton.csv')

# 2.1 Drop Column yang tidak diperlukan
data = df.drop(columns=["id"])

# Mengurangi jumlah data kelas mayoritas agar sesuai dengan jumlah kelas minoritas
data = pd.concat([
    data[data['stroke'] == 0].sample(n=data[data['stroke'] == 1].shape[0], random_state=42),
    data[data['stroke'] == 1]
])

# 2.2 Pisahkan data menjadi fitur dan target
X = data.drop(columns=["stroke"])
y = data["stroke"]

# 2.3 Latih model dengan data yang ada sebelum membuat prediksi
model.fit(X, y)

# Buat prediksi menggunakan model yang sudah dilatih
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# 2.4 Hitung metrik evaluasi
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)

# Tampilkan metrik evaluasi
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.success(f"Accuracy: **{acc:.2f}**")

with col2:
    st.info(f"Precision: **{prec:.2f}**")

with col3:
    st.warning(f"Recall: **{rec:.2f}**")

with col4:
    st.error(f"ROC AUC: **{roc_auc:.2f}**")

# Plot options
plot_option = st.selectbox("Pilih grafik untuk ditampilkan:", ["Pilih", "ROC AUC Curve", "Confusion Matrix"])

if plot_option == "ROC AUC Curve":
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    st.pyplot(fig2)

elif plot_option == "Confusion Matrix":
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

st.markdown("---")
st.subheader("Prediksi Risiko Stroke")

# Input form untuk prediksi
with st.form("prediction_form"):
    gender = st.selectbox("Jenis Kelamin", options=["Male", "Female"])
    age = st.number_input("Umur", min_value=0, max_value=120, value=25)
    hypertension = st.selectbox("Hipertensi", options=["Ya", "Tidak"])
    heart_disease = st.selectbox("Penyakit Jantung", options=["Ya", "Tidak"])
    ever_married = st.selectbox("Pernah Menikah", options=["Ya", "Tidak"])
    work_type = st.selectbox("Jenis Pekerjaan", options=["Pegawai Negeri", "Belum Pernah Bekerja", "Swasta", "Wiraswasta", "Anak-anak"])
    residence_type = st.selectbox("Tempat Tinggal", options=["Rural (Pedesaan)", "Urban (Perkotaan)"])
    avg_glucose = st.number_input("Rata-rata Glukosa", min_value=0.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    smoking_status = st.selectbox("Status Merokok", options=["Tidak diketahui", "Pernah Merokok", "Tidak Pernah Merokok", "Masih Merokok"])

    submit = st.form_submit_button("Prediksi")

# fungsi untuk mapping input ke dalam format yang sesuai dengan model
def map_inputs():
    gender_map = {"Female": 0, "Male": 1}
    hypertension_map = {"Ya": 1, "Tidak": 0}
    heart_map = {"Ya": 1, "Tidak": 0}
    married_map = {"Tidak": 0, "Ya": 1}
    residence_map = {"Rural (Pedesaan)": 0, "Urban (Perkotaan)": 1}
    work_map = {"Pegawai Negeri": 0, "Belum Pernah Bekerja": 1, "Swasta": 2, "Wiraswasta": 3, "Anak-anak": 4}
    smoke_map = {"Tidak diketahui": 0, "Pernah Merokok": 1, "Tidak Pernah Merokok": 2, "Masih Merokok": 3}

    return pd.DataFrame([{
        "gender": gender_map[gender],
        "age": age,
        "hypertension": hypertension_map[hypertension],
        "heart_disease": heart_map[heart_disease],
        "ever_married": married_map[ever_married],
        "work_type": work_map[work_type],
        "Residence_type": residence_map[residence_type],
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "smoking_status": smoke_map[smoking_status]
    }])

if submit:
    input_df = map_inputs()
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Hasil Prediksi")
    st.write(f"Pasien diprediksi: **{'Berpotensi Mengalami Stroke' if pred == 1 else 'Berpotensi Tidak Mengalami Stroke'}**")
    st.write(f"Probabilitas stroke: **{prob:.2f}**")
