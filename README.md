# NLP_TopicModeling

### Introduction
This is an implementation of LDA(Latent Dirichlet allocation) topic modeling as one of my project in QUT IFN619. This project is implemented in Python Jupyternotebook. 

### Data
Dataset used here is 'million ABC news headlines' sources from https://www.kaggle.com/therohk/million-headlines. A million news headlines and its published date are provided in this dataset from 2003 until 2020.

### Packages
```
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
```

### Methodology
To achieve our goal, firstly, a tf/tfidf model will be built based on tri-gram phrases with python scikit-learn library, then a topic modelling technique will be applied. Topic modelling can be described as a method for finding a group of words (i.e topic) from a collection of documents that best represents the information in the collection. In this project, the LDA (Latent Dirichlet allocation ) will be chosen as the techniques for our topic. The evaluation and visualization will be also given.

### FInding and graphs
![topic1](/lda_graphs/topic1.png)
![topic2](/lda_graphs/topic2.png)
![topic3](/lda_graphs/topic3.png)
![topic4](/lda_graphs/topic4.png)
![topic5](/lda_graphs/topic5.png)
![topic6](/lda_graphs/topic6.png)
![topic7](/lda_graphs/topic7.png)
![topic8](/lda_graphs/topic8.png)
![topic9](/lda_graphs/topic9.png)
![topic10](/lda_graphs/topic10.png)
![topic11](/lda_graphs/topic11.png)

![topic_distribution](/lda_graphs/11topics.png)



