import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib

# Датаны зам ба мета өгөгдөл
audio_dataset_path = r"C:\bachelavor\babycry\Baby-Cry-Classification\donateacry-corpus\donateacry_corpus_cleaned_and_updated_data"
metadata = pd.read_csv(r"C:\bachelavor\babycry\Baby-Cry-Classification\full_data.csv", names=['file', 'fold', 'label'])
metadata = metadata.iloc[1:].reset_index(drop=True)

# Metadata-г цэвэрлэх
metadata['fold'] = metadata['fold'].str.replace('fold', '').astype(int)
metadata = metadata[metadata['file'].str.endswith('.wav')]
metadata = metadata.dropna(subset=['fold', 'file', 'label'])
metadata = metadata[metadata['fold'].isin([1, 2, 3, 4, 5])]
metadata['label'] = metadata['label'].str.strip().str.lower()
label_map = {'bp': 'belly_pain', 'bu': 'burping', 'dc': 'discomfort', 'hu': 'hungry', 'ti': 'tired'}
metadata['label'] = metadata['label'].map(label_map)

def augment_audio(audio, sr):
 rate = np.random.uniform(0.8, 1.2)
 audio_stretch = librosa.effects.time_stretch(audio, rate=rate)
 shift = np.random.uniform(-2, 2)
 audio_shift = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)
 noise = np.random.normal(0, 0.005, audio.shape)
 audio_noise = audio + noise
 return [audio_stretch, audio_shift, audio_noise]

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
 if spectrogram_db.shape[1] < target_width:
     spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, target_width - spectrogram_db.shape[1])), mode='constant')
 elif spectrogram_db.shape[1] > target_width:
     spectrogram_db = spectrogram_db[:, :target_width]
 return spectrogram_db

# Датаг бэлтгэх
augmentation_factors = {'belly_pain': 15, 'burping': 31, 'discomfort': 12, 'hungry': 8, 'tired': 15}
data = []
labels = []

for index_num, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
 file_name = os.path.join(audio_dataset_path, f'fold{int(row["fold"])}', row["file"])
 if not os.path.exists(file_name):
     print(f"Файл олдсонгүй, алгаслаа: {file_name}")
     continue
 try:
     audio, sr = librosa.load(file_name)
     label = row["label"]
     aug_factor = augmentation_factors.get(label, 0)
     
     spectrogram = extract_spectrogram(audio, sr)
     data.append(spectrogram[..., np.newaxis])
     labels.append(label)
     
     if aug_factor > 0:
         for _ in range(aug_factor):
             aug_audio = augment_audio(audio, sr)[np.random.randint(0, 3)]
             spectrogram = extract_spectrogram(aug_audio, sr)
             data.append(spectrogram[..., np.newaxis])
             labels.append(label)
 except Exception as e:
     print(f"Алдаа гарлаа: {file_name} - {e}")
     continue

data = np.array(data)
labels = np.array(labels)

# Label encoding
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(labels)

# K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
fold_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(data, label_encoded)):
 print(f"\nFold {fold + 1}/5 эхэллээ...")
 x_train, x_test = data[train_idx], data[test_idx]
 y_train, y_test = label_encoded[train_idx], label_encoded[test_idx]

 # Загвар
 model = Sequential([
     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1), padding='same'),
     BatchNormalization(),
     MaxPooling2D((2, 2)),
     Dropout(0.2),
     Conv2D(64, (3, 3), activation='relu', padding='same'),
     BatchNormalization(),
     MaxPooling2D((2, 2)),
     Dropout(0.2),
     Conv2D(128, (3, 3), activation='relu', padding='same'),
     BatchNormalization(),
     MaxPooling2D((2, 2)),
     Dropout(0.2),
     Conv2D(256, (3, 3), activation='relu', padding='same'),
     BatchNormalization(),
     MaxPooling2D((2, 2)),
     Dropout(0.2),
     Flatten(),
     Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
     Dropout(0.4),
     Dense(5, activation='softmax')
 ])

 model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

 # Callbacks
 early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max')
 reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

 # Сургалт
 history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, 
                     callbacks=[early_stopping, reduce_lr], verbose=1)

 # Үнэлгээ
 y_pred = model.predict(x_test)
 y_pred_classes = np.argmax(y_pred, axis=1)
 print(f"Fold {fold + 1} тайлан:")
 print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
 fold_scores.append(np.mean(y_pred_classes == y_test))

# Моделийг хадгалах (сүүлийн fold-ийн загвар)
model.save("babycry_model_kfold.h5")
joblib.dump(label_encoder, "label_encoder_kfold.pkl")
print(f"Дундаж нарийвчлал: {np.mean(fold_scores) * 100:.2f}%")