import re, time, numpy as np, torch
from collections import Counter
from datasets import load_dataset

t0=time.time()
corpus = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
print('load', time.time()-t0)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[^\w\s]")
def tokenize(text):
    return TOKEN_RE.findall(text.lower())

t0=time.time()
tokens=[]
TARGET=2_000_000
for row in corpus:
    text=row['text']
    if text.strip()=='':
        continue
    tokens.extend(tokenize(text))
    if len(tokens)>=TARGET:
        break
tokens=tokens[:TARGET]
print('tokenize', time.time()-t0, len(tokens))

t0=time.time()
freq=Counter(tokens)
print('count', time.time()-t0, 'vocab_raw', len(freq))

min_count=5
vocab = [w for w,c in freq.items() if c>=min_count]
word2id = {w:i for i,w in enumerate(vocab)}
ids = np.array([word2id[w] for w in tokens if w in word2id], dtype=np.int64)
print('ids', len(ids), 'vocab', len(vocab))

# subsampling
counts = np.array([freq[vocab[i]] for i in range(len(vocab))])
total = counts.sum()
f = counts/total
t_param=1e-5
p_keep = (np.sqrt(f/t_param)+1)*(t_param/f)
p_keep = np.minimum(p_keep,1.0)
rng = np.random.default_rng(42)
keep_mask = rng.random(len(ids)) < p_keep[ids]
ids_sub = ids[keep_mask]
print('after subsample', len(ids_sub), 'kept frac', keep_mask.mean())

def count_pairs(T, window, rng):
    reduced = rng.integers(1, window+1, size=T)
    pos = np.arange(T)
    left = np.minimum(pos, reduced)
    right = np.minimum(T-1-pos, reduced)
    return int((left+right).sum())

def make_pairs(id_arr, window, rng):
    T=len(id_arr)
    reduced = rng.integers(1, window+1, size=T)
    pos = np.arange(T)
    centers_list=[]
    contexts_list=[]
    for d in list(range(-window,0))+list(range(1,window+1)):
        if d<0:
            mask = (reduced>=-d) & (pos+d>=0)
        else:
            mask = (reduced>=d) & (pos+d<T)
        c = pos[mask]
        ctx = c+d
        centers_list.append(c)
        contexts_list.append(ctx)
    centers = np.concatenate(centers_list)
    contexts = np.concatenate(contexts_list)
    return id_arr[centers].astype(np.int32), id_arr[contexts].astype(np.int32)

window=5
t0=time.time()
n_before = count_pairs(len(ids), window, np.random.default_rng(1))
print('count pairs before subsample', n_before, time.time()-t0)

t0=time.time()
c_ids, ctx_ids = make_pairs(ids_sub, window, np.random.default_rng(2))
print('make pairs after subsample', len(c_ids), time.time()-t0)

# negative sampling table
t0=time.time()
unigram_pow = counts.astype(np.float64)**0.75
unigram_pow /= unigram_pow.sum()
table_size=5_000_000
neg_table = rng.choice(len(vocab), size=table_size, p=unigram_pow).astype(np.int32)
print('neg table', time.time()-t0)

# training benchmark
class SGNS(torch.nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.in_emb = torch.nn.Embedding(V, D)
        self.out_emb = torch.nn.Embedding(V, D)
        torch.nn.init.uniform_(self.in_emb.weight, -0.5/D, 0.5/D)
        torch.nn.init.zeros_(self.out_emb.weight)
    def forward(self, centers, contexts, negatives):
        v = self.in_emb(centers)              # B,D
        u_pos = self.out_emb(contexts)         # B,D
        u_neg = self.out_emb(negatives)        # B,K,D
        pos_score = (v*u_pos).sum(-1)
        neg_score = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)  # B,K
        loss = -torch.nn.functional.logsigmoid(pos_score).mean() \
               -torch.nn.functional.logsigmoid(-neg_score).sum(-1).mean()
        return loss

D=100
model = SGNS(len(vocab), D)
opt = torch.optim.Adam(model.parameters(), lr=0.002)
K=5
B=8192
c_t = torch.from_numpy(c_ids.astype(np.int64))
ctx_t = torch.from_numpy(ctx_ids.astype(np.int64))
N = len(c_t)
n_batches = N//B
print('n_batches per epoch', n_batches)

t0=time.time()
perm = torch.randperm(N)
for bi in range(min(n_batches, 200)):
    idx = perm[bi*B:(bi+1)*B]
    cb = c_t[idx]
    ctxb = ctx_t[idx]
    neg_idx = rng.integers(0, table_size, size=(len(cb), K))
    negb = torch.from_numpy(neg_table[neg_idx].astype(np.int64))
    opt.zero_grad()
    loss = model(cb, ctxb, negb)
    loss.backward()
    opt.step()
dt = time.time()-t0
print('200 batches time', dt, 'per batch ms', dt/200*1000)
print('estimated full epoch time (s)', dt/200*n_batches)
