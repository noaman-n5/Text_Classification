import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import string
import nltk
import re
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE

from google.colab import files
files=files.upload()
data = pd.read_csv('MyPersonality - personality_data - MyPersonality - personality_data.csv')

# Preprocessing for data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

data['cleaned_text'] = data['text'].apply(clean_text)

def extract_label(label):
    if pd.isna(label):
        return None
    return label.split(':')[-1]

label_columns = ['O', 'C', 'E', 'A', 'N']
for col in label_columns:
    data[col] = data[col].apply(extract_label)
    data[col] = data[col].map({'y': 1, 'n': 0})

print("Number of rows:", len(data))
print(data.isnull().sum())
print("\nShow Data After clean:")
print(data[['cleaned_text', 'O', 'C', 'E', 'A', 'N']].head(10))
# Add synthetic data to balance all traits
synthetic_texts = [
    # Openness (O)
    ("I love exploring new ideas and trying new experiences.", 1, 0, 0, 0, 0),
    ("I’m always eager to dive into creative projects.", 1, 0, 0, 0, 0),
    # Conscientiousness (C)
    ("I plan everything in advance and keep my life organized.", 0, 1, 0, 0, 0),
    ("I always complete my tasks on time without delays.", 0, 1, 0, 0, 0),
    # Extraversion (E)
    ("I enjoy being around people and feel energized at parties.", 0, 0, 1, 0, 0),
    ("I love talking to everyone and being the center of attention.", 0, 0, 1, 0, 0),
    # Agreeableness (A)
    ("I always help others and avoid conflicts at all costs.", 0, 0, 0, 1, 0),
    ("I make sure everyone around me feels happy and supported.", 0, 0, 0, 1, 0),
    # Neuroticism (N)
    ("I feel anxious all the time and worry about everything.", 0, 0, 0, 0, 1),
    ("I often get overwhelmed by my emotions for no reason.", 0, 0, 0, 0, 1)
]

for col in label_columns:
    print(f"\nDistribution of {col}:")
    print(data[col].value_counts(dropna=False))

# Create a DataFrame for synthetic data
synthetic_data = pd.DataFrame(
    synthetic_texts,
    columns=['cleaned_text', 'O', 'C', 'E', 'A', 'N']
)

# Keep a copy of the original data
data_original = pd.concat([data, synthetic_data], ignore_index=True)
print(f"Data shape after adding synthetic data: {data_original.shape}")
for col in label_columns:
    print(f"Distribution of {col} after synthetic data:")
    print(data_original[col].value_counts(dropna=False))

# === Oversampling Sequentially for ALL Traits ===
for target_trait in label_columns:
    print(f"\nDistribution of {target_trait} before oversampling:")
    print(data[target_trait].value_counts())

    minority_class = data[data[target_trait] == 1]
    majority_class = data[data[target_trait] == 0]

    num_majority = len(majority_class)
    num_minority = len(minority_class)

    if num_minority == 0:
        print(f"No minority class (value=1) found for {target_trait}. Skipping oversampling for this trait.")
        continue

    # Aim for 50/50 balance
    oversample_ratio = max(1, num_majority // num_minority)
    print(f"Oversample ratio for {target_trait}: {oversample_ratio}")

    minority_oversampled = pd.concat([minority_class] * oversample_ratio, ignore_index=True)

    remaining_samples = num_majority - len(minority_oversampled)
    if remaining_samples > 0:
        additional_samples = minority_class.sample(n=remaining_samples, replace=True, random_state=42)
        minority_oversampled = pd.concat([minority_oversampled, additional_samples], ignore_index=True)

    # Update data with the oversampled version
    data = pd.concat([majority_class, minority_oversampled], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nDistribution of {target_trait} after oversampling:")
    print(data[target_trait].value_counts())

# Check final distribution after oversampling
print("\nFinal distribution after sequential oversampling:")
for col in label_columns:
    print(f"\nDistribution of {col}:")
    print(data[col].value_counts(dropna=False))

# Split the data
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
print(f"Total rows: {len(data)}")
print(f"Training rows: {len(train_df)}")
print(f"Testing rows: {len(test_df)}")


trait_columns = ['O', 'C', 'E', 'A', 'N']

# from google.colab import drive
# drive.mount('/content/drive')

# Setup tokenizer and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the dataset class
class PersonalityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.float)
        return item

# Create datasets
train_dataset = PersonalityDataset(
    train_df['cleaned_text'].tolist(),
    train_df[trait_columns].values.tolist(),
    tokenizer
)

test_dataset = PersonalityDataset(
    test_df['cleaned_text'].tolist(),
    test_df[trait_columns].values.tolist(),
    tokenizer
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(trait_columns),
    problem_type="multi_label_classification"
)
model.config.hidden_dropout_prob = 0.4
model.config.attention_probs_dropout_prob = 0.4
model.to(device)

# # Load the saved model weights
# model_path = '/content/drive/MyDrive/Bert_person_improve_retrained_lastVersion#1.pth'
# try:
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     raise

# Training setup
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Continue training the loaded model
for epoch in range(16):
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.numel()
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

# === Test Phase ===
model.eval()
test_loss = 0
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        test_loss += loss.item()

        probabilities = torch.sigmoid(outputs.logits)
        all_probs.append(probabilities.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Concatenate all probabilities and labels
all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Calculate average test loss
avg_test_loss = test_loss / len(test_loader)
print("\n=== Test Phase ===")
print(f"Average Test Loss: {avg_test_loss:.4f}")

# Apply default thresholds to get binary predictions
best_thresholds = [0.5] * len(trait_columns)
test_preds = np.zeros_like(all_probs)
for i, thresh in enumerate(best_thresholds):
    test_preds[:, i] = (all_probs[:, i] > thresh).astype(float)

# Calculate accuracy per trait
print("\nAccuracy per trait:")
for i, trait in enumerate(trait_columns):
    trait_acc = accuracy_score(all_labels[:, i], test_preds[:, i])
    print(f"{trait}: {trait_acc:.4f}")

# Calculate overall accuracy
overall_accuracy = accuracy_score(all_labels.flatten(), test_preds.flatten())
print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")

# === Full Dataset Accuracy ===
print("\n=== Full Dataset Accuracy ===")

# Create dataset for the full data
full_dataset = PersonalityDataset(
    data['cleaned_text'].tolist(),
    data[trait_columns].values.tolist(),
    tokenizer
)

# DataLoader for full dataset
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

# Evaluate on full dataset
model.eval()
full_loss = 0
all_full_probs = []
all_full_labels = []

with torch.no_grad():
    for batch in full_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        full_loss += loss.item()

        probabilities = torch.sigmoid(outputs.logits)
        all_full_probs.append(probabilities.cpu().numpy())
        all_full_labels.append(labels.cpu().numpy())

# Concatenate all probabilities and labels
all_full_probs = np.concatenate(all_full_probs, axis=0)
all_full_labels = np.concatenate(all_full_labels, axis=0)

# Calculate average full dataset loss
avg_full_loss = full_loss / len(full_loader)
print(f"Average Full Dataset Loss: {avg_full_loss:.4f}")

# Apply thresholds to get binary predictions
full_preds = np.zeros_like(all_full_probs)
for i, thresh in enumerate(best_thresholds):
    full_preds[:, i] = (all_full_probs[:, i] > thresh).astype(float)

# Calculate accuracy per trait for full dataset
print("\nAccuracy per trait (Full Dataset):")
for i, trait in enumerate(trait_columns):
    trait_acc = accuracy_score(all_full_labels[:, i], full_preds[:, i])
    print(f"{trait}: {trait_acc:.4f}")

# Calculate overall accuracy for full dataset
overall_full_accuracy = accuracy_score(all_full_labels.flatten(), full_preds.flatten())
print(f"\nOverall Full Dataset Accuracy: {overall_full_accuracy:.4f}")

# === User Input and Personality Analysis ===
def analyze_personality(text, model, tokenizer, trait_columns, device, thresholds):
    model.eval()
    cleaned_text = clean_text(text)
    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()[0]

    probabilities = torch.sigmoid(torch.tensor(logits)).numpy() * 100
    binary_preds = [(probabilities[i] / 100 > thresholds[i]) for i in range(len(trait_columns))]
    result = {
        trait: f"{prob:.2f}% ({'Yes' if binary else 'No'})"
        for trait, prob, binary in zip(trait_columns, probabilities, binary_preds)
    }

# Modified: Select the dominant trait based on the highest probability among "Yes" traits
    if any(binary_preds):
        # Create a list of probabilities, but set to 0 for traits that are "No"
        adjusted_probs = [prob if binary else 0 for prob, binary in zip(probabilities, binary_preds)]
        max_trait_idx = np.argmax(adjusted_probs)
        max_trait = trait_columns[max_trait_idx]
        return result, binary_preds, max_trait
    else:
        return result, binary_preds, None

# Set default thresholds
best_thresholds = [0.5] * len(trait_columns)

# Prompt user for input
print("\nEnter any sentence about yourself, your thoughts, or behavior:")
user_input = input("→ ")
#ده الكود الجديد علشان يغير اختيار السمة المهيمنه بناءا علي اعلي نسبة
try:
    result, binary_preds, max_trait = analyze_personality(user_input, model, tokenizer, trait_columns, device, best_thresholds)

    print("\n🧠 Analyze Your Personality Traits:")
    print("************************************")
    for trait, prediction in result.items():
        print(f"• {trait}: {prediction}")
    print("************************************")

    if max_trait:
        print(f"Your Personality Tends To: {max_trait}")
    else:
        print("No dominant trait detected with current input.")

except Exception as e:
    print(f"Error during prediction: {e}")

# Save the model after retraining
torch.save(model.state_dict(), '/content/drive/MyDrive/Bert_person_improve_retrained_lastVersion#2.pth')
print("Model saved successfully after retraining")