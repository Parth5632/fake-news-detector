import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "final_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# UI
st.title("📰 Fake News Detector")
st.write("Enter a news headline to check if it's Fake or Real.")

user_input = st.text_input("Enter news title here:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: "🟥 Fake", 1: "🟩 Real"}
    st.subheader(f"Prediction: {label_map[prediction]}")
