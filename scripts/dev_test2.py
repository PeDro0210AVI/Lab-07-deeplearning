import time, numpy as np, torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

t0=time.time()
news = load_dataset('fancyzhx/ag_news')
train = news['train']
test = news['test']
print('load news', time.time()-t0)

texts = train['text']
labels = np.array(train['label'])
tr_idx, val_idx = train_test_split(np.arange(len(texts)), test_size=0.1, stratify=labels, random_state=42)
print(len(tr_idx), len(val_idx))

t0=time.time()
vec = TfidfVectorizer(max_features=30000, ngram_range=(1,1))
Xtr = vec.fit_transform([texts[i] for i in tr_idx])
Xval = vec.transform([texts[i] for i in val_idx])
print('tfidf fit', time.time()-t0, Xtr.shape)

t0=time.time()
clf = LogisticRegression(max_iter=200, n_jobs=-1)
clf.fit(Xtr, labels[tr_idx])
pred = clf.predict(Xval)
print('logreg fit+pred', time.time()-t0, 'f1', f1_score(labels[val_idx], pred, average='macro'))

# EmbeddingBag quick test
import re
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[^\w\s]")
def tokenize(text):
    return TOKEN_RE.findall(text.lower())

t0=time.time()
toks = [tokenize(t) for t in texts[:2000]]
print('tokenize 2000 docs', time.time()-t0)
