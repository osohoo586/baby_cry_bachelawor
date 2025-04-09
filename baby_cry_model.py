import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib  # For saving and loading the model
import matplotlib.pyplot as plt

# Path to your audio dataset and metadata
audio_dataset_path = r"C:\bachelavor\babycry\Baby-Cry-Classification\donateacry-corpus\donateacry_corpus_cleaned_and_updated_data"
metadata = pd.read_csv(r"C:\bachelavor\babycry\Baby-Cry-Classification\full_data.csv", names=['file', 'fold', 'label'])

# Feature extraction functions
def feature_1(file):
    audio, sr = librosa.load(file)
    zcr = librosa.zero_crossings(audio)
    zcr = sum(zcr)
    data = pd.DataFrame([zcr], columns=['A'])
    return data

def feature_2(file):
    audio, sr = librosa.load(file)
    mfcc_feature = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=12).T, axis=0)
    mfcc = pd.DataFrame(mfcc_feature)
    mfcc = mfcc.T
    return mfcc

def feature_3(file):
    audio, sr = librosa.load(file)
    chromagram = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512), axis=1)
    cr = pd.DataFrame(chromagram)
    cr = cr.T
    return cr

# Extract features for each audio file
data_1 = pd.DataFrame()  # To store feature_1
data_2 = pd.DataFrame()  # To store feature_2
data_3 = pd.DataFrame()  # To store feature_3

for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path), 'fold' + str(row["fold"]) + '/', str(row["file"]))
    temp_1 = feature_1(file_name)
    temp_2 = feature_2(file_name)
    temp_3 = feature_3(file_name)
    
    # Concatenate data
    data_1 = pd.concat([data_1, temp_1], ignore_index=True)
    data_2 = pd.concat([data_2, temp_2], ignore_index=True)
    data_3 = pd.concat([data_3, temp_3], ignore_index=True)

# Combine all features
result = pd.concat([data_1, data_2, data_3], axis=1, ignore_index=True)
result = result.reset_index(drop=True)

# Extract labels
label = metadata['label']
label = pd.DataFrame(label)

# Combine result features with labels
result_final = pd.concat([result, label], axis=1)

# Align result and label
result, label = result.align(label, axis=0, join='inner')

# Check the shape of the data
print(result.shape)
print(label.shape)

# Feature selection using mutual information
from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(result, label)
mutual_info = pd.Series(mutual_info)
mutual_info.index = result.columns
mutual_info.sort_values(ascending=False)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(result[[0, 3, 24, 23, 10, 6, 16]], label, test_size=0.2, random_state=0)

# Train the model
model = GradientBoostingClassifier()
model.fit(x_train, y_train.values.ravel())

# Calculate accuracy
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy}")

# Predict on the test set
y_pred = model.predict(x_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')

# Show the plot
plt.show()

# Save the model
joblib.dump(model, 'baby_cry_model.pkl')
print("Model saved as 'baby_cry_model.pkl'")

# Load the model for future use
loaded_model = joblib.load('baby_cry_model.pkl')
print("Model loaded successfully!")

# Use the loaded model to make predictions
loaded_accuracy = loaded_model.score(x_test, y_test)
print(f"Accuracy with loaded model: {loaded_accuracy}")




# Zero Crossings (feature_1): Дууны сигналын тэг шилжилтийн тоог тооцоолдог, энэ нь дууны давтамжийн шинж чанарыг илэрхийлдэг.
# Mel-Frequency Cepstral Coefficients (MFCCs, feature_2): Дууны спектрал шинж чанарыг тодорхойлохын тулд MFCC-ийг тооцоолдог, 12 коэффициентийг ашигладаг.
# Chroma Features (feature_3): Дууны хэмнэл ба өнгөний шинж чанарыг тодорхойлохын тулд chroma стфт-ийг тооцоолдог.