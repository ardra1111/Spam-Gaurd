import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')  # Download stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize

    y = []  # Create empty list
    for i in text:
        if i.isalnum():  # Keep alphanumeric characters
            y.append(i)  # Append to list

    processed_text = ""  # Initialize empty string
    for word in y:  # Iterate through words
        if word not in stopwords.words('english') and word not in string.punctuation:
            processed_text += word + " "  # Concatenate

    return [processed_text.strip()]  # Return list with processed text

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Spam Gaurd - SMS Spam Detector')
input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    transform_sms = transform_text(input_sms)  # Preprocess
    vector_input = tfidf.transform(transform_text(input_sms)) # Vectorize
    result = model.predict(vector_input)[0]  # Predict

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
