import streamlit as st
import joblib
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, RegexpStemmer
import re
import string
from string import punctuation

nltk.download('wordnet')
nltk.download('punkt')

model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
# Text Preprocessing Functions
def clean_txt_func1(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,6}\b')
    text = re.sub(pattern, '', text)
    text = re.sub(r"#\w+", '', text)
    text = re.sub(r'@\w+', '', text)
    return text


punctuation = ".,!?;:'\"()[]{}-–/\\|~`"
def clean_txt_func2(text):
    reg_pattern = re.compile(r'(?:https?://|ftp://|www\.)[^\s@]+(?:[:/?#+&;\w-]*|\([^)]*\))?(?:\s|\Z)')
    text = re.sub(reg_pattern, '', text)
    text = ''.join([word for word in text if word not in punctuation])
    words_to_remove = ["subscribe", "subscribers", "like", "comment", "share", "join", "disclaimer", "’"]
    pattern = r'\b(' + '|'.join(words_to_remove) + r')\b|\s*\.\s*'
    text = re.sub(pattern, '', text)
    text = re.sub(r'\b\d+[a-zA-Z]?\b', '', text)
    text = word_tokenize(text)
    return text

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U0001F700-\U0001F77F"
        "\U00002300-\U000023FF"
        "\U00002000-\U000020FF"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001FA00-\U0001FA6F"
        "\U00002B05-\U00002B07"
        "\U00002934-\U00002935"
        "\U00002190-\U000021AA"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def clean_txt_func3(word_list):
    return [remove_emojis(word) for word in word_list if remove_emojis(word).strip()]

def clean_txt_func4(text):
    stopwords_list =[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "did", "didn't",
    "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "isn't", "it", "its", "itself", "just", "ll", "m", "ma", "me",
    "might", "mightn't", "more", "most", "must", "mustn't", "my", "myself", "need",
    "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan't",
    "she", "she's", "should", "shouldn't", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
    "was", "wasn't", "we", "were", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "won't", "would", "y", "you",
    "your", "yours", "yourself", "yourselves"
]

    text = [word for word in text if word not in stopwords_list]
    regexp_stemmer = RegexpStemmer(r'ing$')
    text = [regexp_stemmer.stem(word) for word in text]
    lm = WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text]
    return text

def preprocess_input(title, description):
    title = clean_txt_func1(title)
    description = clean_txt_func1(description)
    title = clean_txt_func2(title)
    description = clean_txt_func2(description)
    title = clean_txt_func3(title)
    description = clean_txt_func3(description)
    title = clean_txt_func4(title)
    description = clean_txt_func4(description)
    combined = title + description
    return ' '.join(combined)

# Streamlit App
st.title("YouTube Video Category Classifier")

# User Input
title = st.text_input("Enter the video title:")
description = st.text_area("Enter the video description:")

if st.button("Predict Category"):
    if title and description:
        # Preprocess the input
        processed_input = preprocess_input(title, description)
        # Vectorize the input
        vectorized_input = tfidf.transform([processed_input])
        # Predict the category
        prediction = model.predict(vectorized_input)
        # Display the prediction
        st.success(f"The predicted category is: {prediction[0]}")
    else:
        st.error("Please provide both a title and a description.")

if st.button("Show Cleaned Text"):
    if title and description:
        # Preprocess the input and display the cleaned text
        cleaned_text = preprocess_input(title, description)
        st.markdown("### Cleaned Text:")
        st.markdown(f"**{cleaned_text}**")
    else:
        st.error("Please provide both a title and a description.")
