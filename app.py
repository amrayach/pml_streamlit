import streamlit as st
from predictExplain import ModelsDeploy
import numpy as np
import pandas as pd



deploy = ModelsDeploy()

st.title('Character-Level CNN Predict & Explain:')
st.text('Select Model:')
model_in = st.selectbox('Models:', ['Yelp-Review-Polarity', 'AG-News-Category-Classifier'], index=0)


sentence = st.text_input('Enter Sentence:', value="Like any Barnes & Noble, it has a nice comfy cafe, and a large selection of books.  The staff is very friendly and helpful.  They stock a decent selection, and the prices are pretty reasonable.  Obviously it's hard for them to compete with Amazon.  However since all the small shop bookstores are gone, it's nice to walk into one every once in a while.")


if model_in == 'Yelp-Review-Polarity':
    prediction, probs = deploy.predict_probs(sentence, model='yelp')
    st.text('--------------------------------')
    if prediction == 0:
        st.text("The Prediction is Negative")
    else:
        st.text("The Prediction is Positive")
    st.text('--------------------------------')
    st.text('Class Probabilities:')

    dataframe = pd.DataFrame(
        np.array([probs]),
        columns=('Negative', 'Positive'))
    st.dataframe(dataframe.style.highlight_max(axis=0))

else:
    prediction, probs = deploy.predict_probs(sentence, model='yelp')
    #st.dataframe(str(prediction))

