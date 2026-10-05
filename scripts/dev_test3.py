import re, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize

news = load_dataset('fancyzhx/ag_news')
train_texts = news['train']['text']
train_labels = np.array(news['train']['label'])

t0=time.time()
train_tok = [word_tokenize(t.lower()) for t in train_texts]
print('tokenize all train', time.time()-t0)

freq = Counter(w for doc in train_tok for w in doc)
vocab = ['<unk>'] + [w for w,c in freq.items() if c>=2]
word2id = {w:i for i,w in enumerate(vocab)}
print('ag vocab size', len(vocab))

def to_ids(doc):
    return [word2id.get(w,0) for w in doc]

ids_docs = [to_ids(d) for d in train_tok]

class BagDataset(torch.utils.data.Dataset):
    def __init__(self, docs, labels):
        self.docs=docs; self.labels=labels
    def __len__(self): return len(self.docs)
    def __getitem__(self, i): return self.docs[i], self.labels[i]

def collate(batch):
    docs, labels = zip(*batch)
    offsets=[0]
    flat=[]
    for d in docs:
        flat.extend(d if len(d)>0 else [0])
        offsets.append(offsets[-1]+max(len(d),1))
    flat_t = torch.tensor(flat, dtype=torch.long)
    offsets_t = torch.tensor(offsets[:-1], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return flat_t, offsets_t, labels_t

idx = np.arange(len(ids_docs))
tr_idx, val_idx = train_test_split(idx, test_size=0.1, stratify=train_labels, random_state=42)

class Clf(nn.Module):
    def __init__(self, vocab_size, dim, n_classes=4, pretrained=None, freeze=False):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, dim, mode='mean')
        if pretrained is not None:
            with torch.no_grad():
                self.emb.weight.copy_(torch.from_numpy(pretrained))
        self.emb.weight.requires_grad = not freeze
        self.mlp = nn.Sequential(nn.Linear(dim,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,n_classes))
    def forward(self, flat, offsets):
        x = self.emb(flat, offsets)
        return self.mlp(x)

model = Clf(len(vocab), 100)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

train_ds = BagDataset([ids_docs[i] for i in tr_idx], train_labels[tr_idx])
val_ds = BagDataset([ids_docs[i] for i in val_idx], train_labels[val_idx])
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False, collate_fn=collate)

t0=time.time()
for epoch in range(2):
    model.train()
    tot_loss=0; n=0
    for flat, offsets, labels in train_dl:
        opt.zero_grad()
        out = model(flat, offsets)
        loss = F.cross_entropy(out, labels)
        loss.backward(); opt.step()
        tot_loss += loss.item()*len(labels); n+=len(labels)
    print('epoch', epoch, 'loss', tot_loss/n, 'time', time.time()-t0)

model.eval()
preds=[]; gts=[]
with torch.no_grad():
    for flat, offsets, labels in val_dl:
        out = model(flat, offsets)
        preds.extend(out.argmax(1).tolist())
        gts.extend(labels.tolist())
print('val f1 macro', f1_score(gts, preds, average='macro'))
print('total time', time.time()-t0)
print('n_params', sum(p.numel() for p in model.parameters()))
