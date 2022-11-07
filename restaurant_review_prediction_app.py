
import pickle
import streamlit as st

# Load the Multinomial Naive Bayes model and TFIDF
classifier = pickle.load(open('/content/drive/MyDrive/Restaurant review prediction /restaurant_review_model.pkl', 'rb'))
tfidf = pickle.load(open('/content/drive/MyDrive/Restaurant review prediction /tfidf-transform.pkl','rb'))

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

st.title("Restaurant Review Sentiment Analyser")
input_text = st.text_area("Enter Your Review")

if st.button('Predict'):

  # Cleaning special character from the sms
  review = re.sub('[^a-zA-Z]', ' ', input_text)

  # Converting the entire sms into lower case
  review = review.lower()

  # Tokenizing the sms by words
  review = review.split()
    
  # Removing the stop words
  filtered_words = [word for word in review if not word in stopwords.words('english')]

  # stemming the words
  stemmed_words = [ps.stem(word) for word in filtered_words]

  # Joining the stemmed words
  review = ' '.join(stemmed_words)

  # vectorize
  vect = tfidf.transform([review])

  # predict
  result = classifier.predict(vect)[0]

  # 4. Display
  if result == 1:
    st.success("Great! POSITIVE review.")
  else:
    st.success("Oops! NEGATIVE review.")
