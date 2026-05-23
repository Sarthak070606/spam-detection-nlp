import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Spam Detection", page_icon="📩")

@st.cache_resource
def load_model():
    model = joblib.load("spam_model.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    return model, tfidf

model, tfidf = load_model()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def cleaning_text_data(text):
    if isinstance(text, list):
        text = text[0]
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]      
    words = [lemmatizer.lemmatize(word) for word in words]          
    return " ".join(words).strip()

st.title("📩 SMS Spam Detection App")
st.write("Enter a message to check whether it is Spam or Not Spam.")

msg = st.text_area("Enter Message")

if st.button("Predict"):
    if not msg.strip():
        st.warning("Please enter a message.")
    else:
        messages = [m for m in msg.split("\n") if m.strip()]
        for single_msg in messages:
            clean_msg = cleaning_text_data(single_msg)
            vector_input = tfidf.transform([clean_msg])
            prediction = model.predict(vector_input)[0]
            if prediction == 1:
                st.error(f"{single_msg} -> Spam")
            else:
                st.success(f"{single_msg} -> Not Spam")

st.markdown("---")
st.markdown("# *:rainbow[*Made with love by* **Sarthak Jain**]* ")