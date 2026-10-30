
import streamlit as st
import joblib
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# Load model files and NLTK resources
@st.cache_resource
def load_resources():

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("extra_trees_model.pkl")
    encoder = joblib.load("label_encoder.pkl")

    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    return vectorizer, model, encoder, stop_words, ps


tfidf, model, encoder, stop_words, ps = load_resources()


# Same preprocessing logic as your notebook
def transform_text(text):

    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Application interface
st.title("📩 SMS Spam Classifier")

st.write(
    "Enter an SMS message below to determine "
    "whether it is Spam or Not Spam."
)

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Type your message here..."
)

if st.button("Check Message", type="primary"):

    if message.strip():

        cleaned_message = transform_text(message)

        vector_input = tfidf.transform([cleaned_message])

        # Your notebook trained using a dense TF-IDF array
        vector_input = vector_input.toarray()

        prediction = model.predict(vector_input)[0]

        result = encoder.inverse_transform([int(prediction)])[0]

        if result == "spam":
            st.error("This message is SPAM!")

        else:
            st.success("This message is Not Spam.")

    else:
        st.warning("Please enter a message first.")
