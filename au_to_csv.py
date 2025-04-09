import os
import pandas as pd

def create_csv_from_audio_files(directory, output_csv):
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return

    print(f"Scanning directory: {directory}")
    print(f"Files found: {os.listdir(directory)}")

    rows = []
    
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if file_name.lower().endswith((".3gp", ".wav")):
            print(f"Processing: {file_name}")

            # Extract label
            parts = file_name.split('-')
            label = parts[-1].split('.')[0] if len(parts) > 1 else "unknown"

            rows.append({
                "name": file_name,
                "folder": os.path.basename(directory),
                "labels": label
            })
        else:
            print(f"Skipping: {file_name}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"CSV created: {output_csv}")
    else:
        print("No audio files found.")

# Run the function

# input_directory = r'C:\bachelavor\babycry\Baby-Cry-Classification\Data\v2\qwerty'
# output_csv = r'C:\bachelavor\babycry\Baby-Cry-Classification\test_data.csv'
input_directory = r'C:\bachelavor\babycry\Baby-Cry-Classification\donateacry-corpus\donateacry_corpus_cleaned_and_updated_data\fold4'
output_csv = r'C:\bachelavor\babycry\Baby-Cry-Classification\full_data4.csv'
# input_directory = r'C:\bachelavor\babycry\Baby-Cry-Classification\donateacry-corpus\donateacry_corpus_cleaned_and_updated_data\fold5'
# output_csv = r'C:\bachelavor\babycry\Baby-Cry-Classification\full_data5.csv'
# input_directory = r'C:\bachelavor\babycry\Baby-Cry-Classification\donateacry-corpus\donateacry_corpus_cleaned_and_updated_data\fold5'
# output_csv = r'C:\bachelavor\babycry\Baby-Cry-Classification\full_data5.csv'
# input_directory = r'C:\bachelavor\babycry\Baby-Cry-Classification\donateacry-corpus\donateacry_corpus_cleaned_and_updated_data\fold5'
# output_csv = r'C:\bachelavor\babycry\Baby-Cry-Classification\full_data5.csv'
create_csv_from_audio_files(input_directory, output_csv)
