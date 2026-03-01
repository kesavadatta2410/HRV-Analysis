import os
import pandas as pd
import json

# 👉 Change this to your folder path
folder_path = r"Data"
eda_report = []

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)

    # Process only CSV and JSON files
    if file.endswith(".csv") or file.endswith(".json"):
        try:
            # Load file
            if file.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:  # JSON
                try:
                    df = pd.read_json(file_path)
                except:
                    # Handle nested JSON
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.json_normalize(data)

            # Get file size in MB
            file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 4)

            # Prepare summary
            file_summary = {
                "filename": file,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "file_size_mb": file_size_mb
            }

            eda_report.append(file_summary)

        except Exception as e:
            print(f"Error processing {file}: {e}")

# Save JSON report
output_path = os.path.join(folder_path, "EDA_Report.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(eda_report, f, indent=4)

print(f"\n✅ EDA Report saved at: {output_path}")