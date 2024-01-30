import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')  # Download stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize

    processed_text = []  # Create empty list to store processed words
    for word in text:
        if word.isalnum():  # Keep alphanumeric characters
            word = ps.stem(word)  # Apply stemming
            if word not in stopwords.words('english') and word not in string.punctuation:
                processed_text.append(word)  # Append processed word to list

    return " ".join(processed_text)  # Return processed text as a string

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Spam Gaurd - SMS Spam Detector')
input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    processed_sms = transform_text(input_sms)  # Preprocess
    vector_input = tfidf.transform([processed_sms])  # Vectorize (note the list format)
    result = model.predict(vector_input)[0]  # Predict

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

