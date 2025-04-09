import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import classification_report

# Загвар ба label encoder-г ачаалах
model = load_model("babycry_model_kfold.h5")
label_encoder = joblib.load("label_encoder_kfold.pkl")

def preprocess_audio(audio, sr, target_length=6):
    target_samples = sr * target_length
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    return audio

def extract_spectrogram(audio, sr, target_width=128):
    audio = preprocess_audio(audio, sr)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=512)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    # Хэмжээг тогтмол (128, 128) болгох
    if spectrogram_db.shape[1] < target_width:
        spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, target_width - spectrogram_db.shape[1])), mode='constant')
    elif spectrogram_db.shape[1] > target_width:
        spectrogram_db = spectrogram_db[:, :target_width]  # Илүү урт хэсгийг таслах
    return spectrogram_db

def test_folder_audios(folder_path, target_width=128):
    predicted_labels = []
    true_labels = []
    correct = 0
    total_labeled = 0

    # Загварын сурсан классууд ба товчлолын толь
    valid_labels = set(label_encoder.classes_)  # {'belly_pain', 'burping', 'discomfort', 'hungry', 'tired'}
    label_short_map = {'bp': 'belly_pain', 'bu': 'burping', 'dc': 'discomfort', 'hu': 'hungry', 'ti': 'tired'}

    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                audio, sr = librosa.load(file_path)
                spectrogram = extract_spectrogram(audio, sr, target_width=target_width)
                spectrogram = spectrogram[..., np.newaxis]  # (128, 128, 1)
                spectrogram = np.expand_dims(spectrogram, axis=0)  # (1, 128, 128, 1)

                prediction = model.predict(spectrogram)
                predicted_class = np.argmax(prediction, axis=1)[0]
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]

                # Файлын нэрнээс жинхэнэ шошгыг авах
                true_label = None
                filename_lower = filename.lower()
                for label in valid_labels:
                    if label in filename_lower:
                        true_label = label
                        break
                if true_label is None:
                    for short, full in label_short_map.items():
                        if short in filename_lower:
                            true_label = full
                            break
                if true_label is None:
                    true_label = "Тодорхойгүй"

                print(f"Файл: {filename}, Таамагласан: {predicted_label}, Жинхэнэ: {true_label}")

                # Зөвхөн тодорхой шошготой файлуудыг тооцох
                if true_label != "Тодорхойгүй":
                    true_labels.append(true_label)
                    predicted_labels.append(predicted_label)
                    if true_label == predicted_label:
                        correct += 1
                    total_labeled += 1

            except Exception as e:
                print(f"Алдаа: {file_path} - {e}")
                continue

    # Нарийвчлалыг хэвлэх
    if total_labeled > 0:
        accuracy = (correct / total_labeled) * 100
        print(f"\nТодорхой шошготой файлуудын нарийвчлал: {accuracy:.2f}% (Зөв: {correct}/{total_labeled})")
    else:
        print("Тодорхой шошготой файл олдсонгүй.")

    # Classification report-г хэвлэх
    if true_labels and predicted_labels:
        print("\nТайлан:")
        print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_, labels=list(valid_labels)))
    else:
        print("Тайлан гаргахад хангалттай мэдээлэл алга.")

# Тест хавтасыг шалгах
folder_path = r"C:\bachelavor\babycry\Baby-Cry-Classification\DATA\v2\qwerty"  # Таны тест хавтас
test_folder_audios(folder_path, target_width=128)