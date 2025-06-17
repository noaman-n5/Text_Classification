# Text_Classification
i used Bert model and framework PyTorch for text classification , to analyze personality traits
#🧠 Personality Trait Detection using BERT
This project uses **BERT (Bidirectional Encoder Representations from Transformers)** for multi-label classification to detect Big Five personality traits (OCEAN model) from text input.

## 🔍 Objective
Build a robust NLP pipeline that:
- Cleans and preprocesses user-generated text
- Balances imbalanced trait labels using synthetic data and oversampling
- Trains a multi-label classification model using `bert-base-uncased`
- Analyzes and outputs personality trait probabilities
- Predicts the **dominant personality trait** based on the highest probability

## 🧪 Personality Traits (OCEAN)
- **O**: Openness
- **C**: Conscientiousness
- **E**: Extraversion
- **A**: Agreeableness
- **N**: Neuroticism

## 📁 Dataset
Upload your CSV file with the following format:

| text | O | C | E | A | N |
|------|---|---|---|---|---|
|"text"| y | n | y | n | y |

### Notes:
- The labels must be strings like `O:y`, `C:n`, etc.
- The script will clean and extract the binary labels from this format.
---

## ⚙️ Setup Instructions and training model
### 1. Install Dependencies
```bash
pip install torch transformers nltk pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
2. Download NLTK Resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
3. Upload Dataset (in Google Colab)
from google.colab import files
files.upload()

##🧼Data Preprocessing
Clean and tokenize text
Remove punctuation, digits
Add synthetic balanced samples per trait
Apply oversampling sequentially for each trait to balance class distribution

🤖 Model Architecture and hyperparameter
BertForSequenceClassification with 5 output nodes for O, C, E, A, N
Multi-label classification using BCEWithLogitsLoss
Optimizer: AdamW
Dropout: 0.4
Trained for 16 epochs

📊 Evaluation
Calculates accuracy per trait
Calculate Evaluation Metric
Prints test and full dataset accuracy
Uses threshold of 0.5 for classification


##🧠 Personality Analysis from User Input
The final section allows users to input a sentence.
--The model:
 Preprocesses the text
 Predicts trait probabilities
 Identifies dominant personality trait (highest probability among positive predictions)

💾 Save & Load Model
# Save
torch.save(model.state_dict(), 'model_path.pth')
# Load
model.load_state_dict(torch.load('model_path.pth'))

🧪 Example
text:
→ I love trying new experiences and exploring different cultures.
🧠 Analyze Your Personality Traits:
************************************
• O: 85.13% (Yes)
• C: 33.29% (No)
• E: 58.92% (Yes)
• A: 42.17% (No)
• N: 28.94% (No)
************************************
Your Personality Tends To: O
📁 File Structure
├── dataset.csv
├── personality_analysis.ipynb
├── model.pth
├── README.md
📌 Future Improvements
Use a fine-tuned BERT variant (e.g., RoBERTa)
Add user interface with Streamlit or Flask
Improve dominant trait logic using fuzzy thresholds or dynamic cutoffs
Evaluate with additional personality datasets

🧑‍💻 Author
Developed with ❤️ using PyTorch and HuggingFace by [Abdulrahman noaman]
