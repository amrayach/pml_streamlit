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


_html = {}
RENDER_WRAPPER = None

def render_old(
    docs, style="dep", page=False, minify=False, jupyter=None, options={}, manual=False
):
    """Render displaCy visualisation.

    docs (list or Doc): Document(s) to visualise.
    style (unicode): Visualisation style, 'dep' or 'ent'.
    page (bool): Render markup as full HTML page.
    minify (bool): Minify HTML markup.
    jupyter (bool): Override Jupyter auto-detection.
    options (dict): Visualiser-specific options, e.g. colors.
    manual (bool): Don't parse `Doc` and instead expect a dict/list of dicts.
    RETURNS (unicode): Rendered HTML markup.

    DOCS: https://spacy.io/api/top-level#displacy.render
    USAGE: https://spacy.io/usage/visualizers
    """
    factories = {
        "dep": (DependencyRenderer, parse_deps),
        "ent": (EntityRenderer, parse_ents),
    }
    if style not in factories:
        raise ValueError(Errors.E087.format(style=style))
    if isinstance(docs, (Doc, Span, dict)):
        docs = [docs]
    docs = [obj if not isinstance(obj, Span) else obj.as_doc() for obj in docs]
    if not all(isinstance(obj, (Doc, Span, dict)) for obj in docs):
        raise ValueError(Errors.E096)
    renderer, converter = factories[style]
    renderer = renderer(options=options)
    parsed = [converter(doc, options) for doc in docs] if not manual else docs
    _html["parsed"] = renderer.render(parsed, page=page, minify=minify).strip()
    html = _html["parsed"]
    if RENDER_WRAPPER is not None:
        html = RENDER_WRAPPER(html)

    return html

#os.system('python -m spacy download de_core_news_sm')

nlp = spacy.load("en_core_web_sm")

spacy.displacy.render = render_old
deploy = ModelsDeploy()

st.title('Character-Level CNN Predict & Explain:')
st.text('Select Model:')
model_in = st.selectbox('Models:', ['Yelp-Review-Polarity', 'AG-News-Category-Classifier'], index=0)


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
    words = list(filter(lambda x: x != '', list(heatmap.keys())))
    spaces = [True] * (len(words) - 1)
    spaces.append(False)
    doc = Doc(nlp.vocab, words=words, spaces=spaces)

    ents = []
    for j, i in enumerate(doc):
        new_ent = Span(doc, j, j + 1, label=str(j))
        ents.append(new_ent)


    doc.ents = []
    doc.ents = ents



    tags = ['0', '2', '3', '4', '5', 'State_of_health',
            'Process', 'Medication', 'Time_information', 'Local_specification', 'Biological_chemistry',
            'Biological_parameter', 'Dosing', 'Person', 'Medical_specification', 'Medical_device', 'Body_Fluid',
            'Degree', 'Tissue']
    colors = ['#E8DAEF', '#85C1E9', '#FAD7A0', '#ABEBC6', '#F7DC6F', '#F9E79F', '#A9DFBF', '#7FB3D5', '#F5B041',
              '#AED6F1', '#82E0AA', '#F4D03F', '#58D68D', '#A2D9CE', '#F8C471', '#D2B4DE', '#D7BDE2', '#76D7C4',
              '#614ec2', '#f59b47']
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

    with open('sample_heatmap.pdf', "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)




