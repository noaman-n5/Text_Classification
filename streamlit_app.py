import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Setup device and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=5, 
    problem_type="multi_label_classification"
)
model.config.hidden_dropout_prob = 0.4
model.config.attention_probs_dropout_prob = 0.4

# Load the saved model weights
model_path = 'D:/Graduation Project/Final code Bert/model lastVersion#2/Bert_person_improve_retrained_lastVersion#2.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    st.success("Model Loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise

# Define trait columns and full names
trait_columns = ['O', 'C', 'E', 'A', 'N']
trait_full_names = {
    'O': 'Openness(O)',
    'C': 'Conscientiousness(C)',
    'E': 'Extraversion(E)',
    'A': 'Agreeableness(A)',
    'N': 'Neuroticism(N)'
}
best_thresholds = [0.5] * len(trait_columns)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Personality analysis function
def analyze_personality(text, model, tokenizer, trait_columns, trait_full_names, device, thresholds):
    try:
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
            trait_full_names[trait]: f"{prob:.2f}%"
            for trait, prob in zip(trait_columns, probabilities)
        }
        
        dominant_trait = None
        if any(binary_preds):
            # Select the dominant trait based on the highest probability among "Yes" traits
            adjusted_probs = [prob if binary else 0 for prob, binary in zip(probabilities, binary_preds)]
            max_trait_idx = np.argmax(adjusted_probs)
            dominant_trait = trait_full_names[trait_columns[max_trait_idx]]
        
        return result, dominant_trait
    except Exception as e:
        raise Exception(f"Error Of Prediction: {str(e)}")

# Streamlit UI
st.title("🧠 Personality Traits Analyst")
st.markdown("""
Enter a sentence about yourself, your thoughts, or your behavior, and discover your personality traits based on the Big Five model!
""")

# Text input
user_input = st.text_area("Enter Text:")

# Analysis button
if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please Enter A Valid Sentence")
    else:
        try:
            # Perform personality analysis
            result, dominant_trait = analyze_personality(
                user_input, model, tokenizer, trait_columns, trait_full_names, device, best_thresholds
            )

            # Display results
            st.subheader("Personality Analysis:")
            st.markdown("***")
            for trait, prob in result.items():
                st.write(f"**{trait}**: {prob}")
            st.markdown("***")

            st.subheader("Dominant Trait:")
            st.write(dominant_trait if dominant_trait else "No Dominant Trait Detected.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("""
---
*Powered by BERT and Streamlit | Built for personality analysis based on the Big Five*
""")