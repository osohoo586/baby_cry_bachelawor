import numpy as np
import pandas as pd
import os
import joblib
import librosa

def feature_1(file):
    audio, sr = librosa.load(file)
    zcr = librosa.zero_crossings(audio)
    zcr = sum(zcr)
    return pd.DataFrame([zcr], columns=[0])

def feature_2(file):
    audio, sr = librosa.load(file)
    mfcc_feature = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=12).T, axis=0)
    return pd.DataFrame([mfcc_feature], columns=list(range(1, 13)))

def feature_3(file):
    audio, sr = librosa.load(file)
    chromagram = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512), axis=1)
    return pd.DataFrame([chromagram], columns=list(range(13, 25)))

def classify_new_file(file_path):
    feat1 = feature_1(file_path)
    feat2 = feature_2(file_path)
    feat3 = feature_3(file_path)
    
    combined_features = pd.concat([feat1, feat2, feat3], axis=1)
    selected_columns = [0, 3, 24, 23, 10, 6, 16]
    input_data = combined_features[selected_columns]
    
    model = joblib.load('baby_cry_model.pkl')
    prediction = model.predict(input_data)
    
    return prediction[0]

new_file_path = r"C:\bachelavor\babycry\Baby-Cry-Classification\Data\v2\qwerty\generated1_5f651112-d2a3-4911-b3a5-f37bfc092494-1430734603484-1.7-m-72-sc.wav"
label = classify_new_file(new_file_path)
print(f"Тодорхойлогдсон ангилал: {label}")




# Zero Crossings (feature_1): Дууны сигналын тэг шилжилтийн тоог тооцоолдог, энэ нь дууны давтамжийн шинж чанарыг илэрхийлдэг.
# Mel-Frequency Cepstral Coefficients (MFCCs, feature_2): Дууны спектрал шинж чанарыг тодорхойлохын тулд MFCC-ийг тооцоолдог, 12 коэффициентийг ашигладаг.
# Chroma Features (feature_3): Дууны хэмнэл ба өнгөний шинж чанарыг тодорхойлохын тулд chroma стфт-ийг тооцоолдог.