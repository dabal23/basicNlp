import pandas as pd
import joblib
from sklearn.svm import SVC
import streamlit as st
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

# create a function to extract part of speech (POS)


def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'V': wordnet.VERB,
                'N': wordnet.NOUN,
                'J': wordnet.ADJ,
                'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


model = joblib.load('trained_model.joblib')
bow_vectorizer = joblib.load('count_vectorizer.pkl')


def clean_text(text):
    text_clean = re.sub(r"http\S+", '', text)
    text_clean = re.sub(r"http", '', text_clean)
    text_clean = re.sub(r"@\S+", '', text_clean)
    text_clean = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", ' ', text_clean)
    text_clean = re.sub(r"@", "at", text_clean)
    text_clean = text_clean.replace('#', '')
    text_clean = text_clean.lower()
    return text_clean


st.header('Tweet disaster detection')
st.write('This project is one of our assigment to do nlp related project. This is detecting wheter the tweet is disaster or not. This model is using bag of words vectorizer and SVM model')


st.subheader('How to use?')
st.write('Paste the tweet in the textbox or write anything on text box, then click enter')
# Create a text input widget
user_input = st.text_input(
    "Enter some text:", "Paste your tweet or write anything here")
user_input = clean_text(user_input)

user_input = word_tokenize(user_input)
user_input = [i for i in user_input if i not in stopwords.words('english')]
user_input = [lemmatizer.lemmatize(i, get_pos(i)) for i in user_input]
user_input = [" ".join(user_input)]

data = bow_vectorizer.transform(user_input)
predict = model.predict(data)
status = ['That sentence is disaster related sentence' if i ==
          1 else 'That sentence is NOT disaster related sentence' for i in predict]


# Display the entered text
st.write("You entered:", user_input[0])
st.write("Status:", status[0])


dataset = open('dataset.png', 'rb').read()
distribution = open('distribution label.png', 'rb').read()
cloudword = open('cloud word.png', 'rb').read()
length = open('tweet length.png', 'rb').read()
model = open('model comparison.png', 'rb').read()

# Display the image within an expander
with st.expander("About the model"):
    st.write('The dataset is from kaggle, consist of tweet data and its label')
    st.image(dataset, caption='Dataset', use_column_width=True)
    st.write(
        'Label distribution shows that label distribution seems relatively balance')
    st.image(distribution, caption='Dataset', use_column_width=True)
    st.write('Cloud words based on each labels')
    st.image(cloudword, caption='Dataset', use_column_width=True)
    st.write('Tweet length distribution for disaster and non disaster. Disaster tweet tend to have longer tweet than non disaster')
    st.image(length, caption='Dataset', use_column_width=True)
    st.write('Model comparison and vectorization method comparison')
    st.image(model, caption='Dataset', use_column_width=True)
