
# 🧪 Sentiment Analysis Lab: Classifying Text Using Machine Learning

## Overview
This lab is based on **Chapter 8** of *Python Machine Learning (2nd Edition)* by Sebastian Raschka and Vahid Mirjalili.  
The chapter demonstrates how to apply machine learning to **sentiment analysis**—predicting opinions (positive or negative) from text data using natural language processing (NLP) techniques.

In this hands-on lab, you will:

- Prepare and clean real-world text data (IMDb movie reviews)
- Transform text into numerical features using the **Bag-of-Words** and **TF-IDF** models  
- Build and evaluate a **Logistic Regression** classifier  
- Implement **Out-of-Core Learning** for large text datasets  
- Perform **Topic Modeling** using **Latent Dirichlet Allocation (LDA)**

---

## Objectives

By the end of this lab, you should be able to:

✅ Understand the NLP preprocessing workflow (tokenization, stop words, stemming)  
✅ Build feature vectors using **CountVectorizer** and **TfidfVectorizer**  
✅ Train and tune a logistic regression model for sentiment prediction  
✅ Apply **incremental learning** using `SGDClassifier` for large datasets  
✅ Use **LDA** for unsupervised topic discovery  

---

## Prerequisites

- **Python 3.x**
- **Libraries:**  
  `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `nltk`, `pyprind`

To install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib nltk pyprind
```

---

## Part 1: Preparing and Cleaning Text Data

### Step 1.1 — Download IMDb Movie Review Dataset
```python
import pyprind
import pandas as pd
import os

basepath = 'aclImdb'
labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()

for s in ('train', 'test'):
    for label in ('pos', 'neg'):
        path = os.path.join(basepath, s, label)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
                df = df.append([[txt, labels[label]]], ignore_index=True)
                pbar.update()

df.columns = ['review', 'sentiment']
df.to_csv('movie_data.csv', index=False, encoding='utf-8')
```

### Step 1.2 — Clean Text with Regex and Emoticons
```python
import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

df['review'] = df['review'].apply(preprocessor)
```

---

## Part 2: Building Feature Vectors

### Step 2.1 — Create a Bag-of-Words Representation
```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet'
])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())
```

### Step 2.2 — Apply TF-IDF Transformation
```python
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())
```

---

## Part 3: Training a Sentiment Classifier

### Step 3.1 — Tokenization, Stop Words, and Lemmatization
```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')

def tokenizer(text):
    return text.split()
```

### Step 3.2 — Train Logistic Regression with Grid Search
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [{
    'vect__ngram_range': [(1, 1)],
    'vect__stop_words': [stop, None],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0]
}]

lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(random_state=0))
])

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1)
gs_lr_tfidf.fit(X_train, y_train)
print('Best Accuracy: %.3f' % gs_lr_tfidf.best_score_)
print('Best Params:', gs_lr_tfidf.best_params_)
```

---

## Part 4: Out-of-Core Learning for Large Datasets

### Step 4.1 — Stream Data from Disk
```python
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y
```

### Step 4.2 — Train Incrementally with SGDClassifier
```python
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, max_iter=1)
doc_stream = stream_docs(path='movie_data.csv')

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
```

---

## Part 5: Topic Modeling Using Latent Dirichlet Allocation (LDA)
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')

count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
X = count.fit_transform(df['review'].values)

lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)

n_top_words = 5
feature_names = count.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
```

---

## Analysis Questions
1. Why is it important to remove HTML tags and punctuation before modeling?  
2. How do emoticons contribute to sentiment analysis?  
3. Compare logistic regression performance using different vectorizers.  
4. What problem does out-of-core learning solve?  
5. What do LDA topics represent?  

---

## Deliverables
  
- Visualizations for TF-IDF and topic modeling  
- Written answers to analysis questions  
- Summary report (max 1 page)  

---

## Submission
Push your completed lab to GitHub with:
```
README.md
movie_data.csv
sentiment_lab.ipynb
visualizations/
```
