# NLP_TopicModeling

### Introduction
This is an implementation of LDA(Latent Dirichlet allocation) topic modeling as one of my project in QUT IFN619. This project is implemented in Python Jupyternotebook. 

### Data
Dataset used here is 'million ABC news headlines' sources from https://www.kaggle.com/therohk/million-headlines. A million news headlines and its published date are provided in this dataset from 2003 until 2020.

### Packages
import pandas as pd               
import matplotlib.pyplot as plt  
import datetime
import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
import re 
import nltk 
import string
from gensim.models import CoherenceModel
from IPython.display import Image
from matplotlib import pyplot as plt
from wordcloud import WordCloud

### Methodology
To achieve our goal, firstly, a tf/tfidf model will be built based on tri-gram phrases with python scikit-learn library, then a topic modelling technique will be applied. Topic modelling can be described as a method for finding a group of words (i.e topic) from a collection of documents that best represents the information in the collection. In this project, the LDA (Latent Dirichlet allocation ) will be chosen as the techniques for our topic. The evaluation and visualization will be also given.
