import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Load the data with correct encoding
try:
    # First try UTF-8
    df = pd.read_csv('DataSource\\Bank_data.csv')
except UnicodeDecodeError:
    # If UTF-8 fails, try common alternatives
    try:
        df = pd.read_csv('DataSource\\Bank_data.csv', encoding='latin1')
    except:
        df = pd.read_csv('DataSource\\Bank_data.csv', encoding='windows-1252')

# Check if the data loaded correctly
print("Data loaded successfully. First few rows:")
print(df.head())
print(f"Total samples: {len(df)}")

# Format for fine-tuning (ChatML format for better results)
def format_data(row):
    return {
        "text": f"<|im_start|>user\n{row['Query']}<|im_end|>\n<|im_start|>assistant\n{row['Response']}<|im_end|>"
    }

formatted_data = [format_data(row) for _, row in df.iterrows()]

# Split data
train_data, val_data = train_test_split(formatted_data, test_size=0.2, random_state=42)

# Save as JSONL
def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

save_jsonl(train_data, 'train.jsonl')
save_jsonl(val_data, 'val.jsonl')

print(f"Data preparation complete. Train: {len(train_data)}, Val: {len(val_data)} samples")
