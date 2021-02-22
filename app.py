import base64
import os
import streamlit as st
from predictExplain import ModelsDeploy
import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Span
import streamlit as st
from typing import Optional
from spacy import displacy
from spacy.displacy import DependencyRenderer, EntityRenderer
from spacy.tokens import Doc, Span
from spacy.errors import Errors, Warnings
import warnings
import re
from spacy_streamlit.util import get_svg
from spacy.displacy import parse_ents, parse_deps
from spacy_streamlit import visualize_ner

def to_rgba(hex, val):
    val = int(val * 20)
    val = abs(val)
    val = 255 if val > 255 else val
    hex = hex + "{:02x}".format(val)
    return hex

nlp = spacy.load("en_core_web_sm")

deploy = ModelsDeploy()

st.title('Character-Level CNN Predict & Explain:')
st.text('Select Model:')
model_in = st.selectbox('Models:', ['Yelp-Review-Polarity', 'AG-News-Category-Classifier'], index=1)


sentence = st.text_input('Enter Sentence:', value="Like any Barnes & Noble, it has a nice comfy cafe, and a large selection of books.  The staff is very friendly and helpful.  They stock a decent selection, and the prices are pretty reasonable.  Obviously it's hard for them to compete with Amazon.  However since all the small shop bookstores are gone, it's nice to walk into one every once in a while.")

if model_in == 'Yelp-Review-Polarity':
    prediction, probs, heatmap = deploy.explain(sentence, model='yelp')
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

    #doc = nlp("Some text")
    #ents = list(doc.ents)
    words = [i[0] for i in heatmap]
    vals = [i[1] for i in heatmap]
    spaces = [True] * (len(words) - 1)
    spaces.append(False)
    doc = Doc(nlp.vocab, words=words, spaces=spaces)

    ents = []
    tags = []
    for j, i in enumerate(doc):
        new_ent = Span(doc, j, j + 1, label=str(j))
        ents.append(new_ent)
        tags.append(str(j))

    doc.ents = []
    doc.ents = ents

    sum = sum(vals)
    vals = [v*100/sum for v in vals]

    col_library = {'positive': '#FF0000', 'negative': '#0000FF'}
    colors = [to_rgba(col_library['positive'], x) if x >= 0 else to_rgba(col_library['negative'], x) for x in vals]

    tags = tuple(list(map(lambda x: ''.join(list(map(lambda y: y.upper(), x.split('_')))), tags)))
    col_dict = {}
    for i in range(len(tags)):
        col_dict[tags[i]] = colors[i]

    visualize_ner(doc, labels=tags, colors=col_dict)

    #with open('sample_heatmap.pdf', "rb") as f:
    #    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    #pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    #st.markdown(pdf_display, unsafe_allow_html=True)
    #st.latex(open('basic.tex', encoding='utf8').read())

else:
    prediction, probs, heatmap = deploy.explain(sentence, model='ag_news')

    st.text('--------------------------------')
    if prediction == 0:
        st.text("The Prediction is World")
    elif prediction == 1:
        st.text("The Prediction is Sports")

    elif prediction == 2:
        st.text("The Prediction is Business")

    else:
        st.text("The Prediction is Sci/Tech")
    st.text('--------------------------------')
    st.text('Class Probabilities:')

    dataframe = pd.DataFrame(
        np.array([probs]),
        columns=('World', 'Sports', 'Business', 'Sci/Tech'))
    st.dataframe(dataframe.style.highlight_max(axis=0))

    words = [i[0] for i in heatmap]
    vals = [i[1] for i in heatmap]
    spaces = [True] * (len(words) - 1)
    spaces.append(False)
    doc = Doc(nlp.vocab, words=words, spaces=spaces)

    ents = []
    tags = []
    for j, i in enumerate(doc):
        new_ent = Span(doc, j, j + 1, label=str(j))
        ents.append(new_ent)
        tags.append(str(j))

    doc.ents = []
    doc.ents = ents

    sum = sum(vals)
    vals = [v * 100 / sum for v in vals]
    threshold = np.mean(vals)
    print(heatmap)
    col_library = {'positive': '#FF0000', 'negative': '#0000FF'}
    colors = [to_rgba(col_library['positive'], x) if x >= threshold else to_rgba(col_library['negative'], x) for x in vals]

    tags = tuple(list(map(lambda x: ''.join(list(map(lambda y: y.upper(), x.split('_')))), tags)))
    col_dict = {}
    for i in range(len(tags)):
        col_dict[tags[i]] = colors[i]

    visualize_ner(doc, labels=tags, colors=col_dict)





