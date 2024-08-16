import torch
from torch import nn
import tiktoken
import math
from torch.utils.data import Dataset, DataLoader
import urllib.request
import numpy as np
# url = (
#     "https://raw.githubusercontent.com/rasbt/"
#     "LLMs-from-scratch/main/ch05/"
#     "01_main-chapter-code/gpt_download.py"
# )
# filename = url.split('/')[-1]
# urllib.request.urlretrieve(url, filename)
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
from MY_GPT.gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
with open("t1.txt", "r", encoding="utf-8") as f:
    raw_text=f.read()
# processing_1=re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# processing_2=[i.strip() for i in processing_1 if i.strip()]#strip() and split() do not mix them up
# words_list=sorted(set(processing_2))
# words_dict_1_raw={token:number for number,token in enumerate(words_list)}
# words_dict_2_raw={number:token for number,token in enumerate(words_list)}
# processed = words_list + ["<|endoftext|>", "<|unk|>"]
# words_dict_1={token:number for number,token in enumerate(processed)}
# words_dict_2={number:token for number,token in enumerate(processed)}
# class tokenizerv1:
#     def __init__(self):
#         self.words_dict_1 = words_dict_1
#         self.words_dict_2 = words_dict_2
#     def tokentoid(self,token_input):
#         processing = re.split(r'([,.:;?_!"()\']|--|\s)', token_input)
#         processing = [item.strip() for item in processing if item.strip()]
#         processing = [item if item in word_list
#                       else "<|unk|>" for item in processing]
#         ids=[self.words_dict_1[token] for token in processed ]
#         return ids
#     def idtotoken(self,id_input):
#         tokens=" ".join([self.words_dict_2[number] for number in id_input])
#         tokens=re.sub(r'\s+([,.?!"()\'])',r'\1',tokens)
#         return tokens

# GPT_CONFIG_124M = {
#     "vocab_size": 50257,  # Vocabulary size
#     "context_length": 1024,      # Context length
#     "emb_dim": 768,       # Embedding dimension
#     "n_heads": 12,        # Number of attention heads
#     "n_layers": 12,       # Number of layers
#     "drop_rate": 0.1,     # Dropout rate
#     "qkv_bias": False     # Query-Key-Value bias
# }
tokenizer = tiktoken.get_encoding("gpt2")
text_data=tokenizer.encode(raw_text)
# ids = tokenizer.encode(raw_text)
# enc_sample = enc_text[50:]
# context_size=4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
# for i in range(1,context_size+1):
#     data=enc_sample[:i]
#     pred=enc_sample[i]
#     print(f"train: {data}, prediction: {pred}")
class dataset(Dataset):
    def __init__(self,ids,max_len,stride):
        self.ids_data=[]
        self.ids=ids
        self.ids_pred=[]
        for i in range(0,len(ids)-max_len,stride):
            self.ids_data.append(torch.tensor(ids[i:i+max_len]))
            self.ids_pred.append(torch.tensor(ids[i+1:i+max_len+1]))

    # def printing(self):
    #     print(f"{self.ids_data}\n\n\n\n\n\n{self.ids_pred}")
    def __getitem__(self,index):
        return self.ids_data[index],self.ids_pred[index]
    def __len__(self):
        return len(self.ids_data)

# dataset1=dataset(ids=ids[0:500],max_len=256,stride=128)
# # dataset1.printing()
def create_dataloader_v1(ids,batch_size=4, max_length=256,
               stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset1=dataset(ids=ids,max_len=max_length,stride=stride)
    dataloader = DataLoader(
        dataset1,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0
    )
    return dataloader

# dataloader1=dataloader(ids=ids,batch_size=8, max_length=4, stride=1, shuffle=False, drop_last=True, num_workers=0)
# data_iter=iter(dataloader1)
# inputs, preds=next(data_iter)
# vocab_size = 50257
# output_dim = 256
# torch.manual_seed(123)
# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# token_embedding=token_embedding_layer(inputs)
# context_length = max_length=4
# pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# input_embeddings=pos_embeddings+token_embedding
# class attentionblock(nn.Module):
#     def __init__(self,d_in,d_out,sequence_length,dropout_prob,num_head,qkv_bias=False):
#         super().__init__()
#         self.d_in =d_in
#         self.d_out=d_out
#         self.sequence_length=sequence_length
#         self.num_head=num_head
#         self.d_head=self.d_out//self.num_head
#         self.q=nn.Linear(d_in,d_out)
#         self.k=nn.Linear(d_in,d_out)
#         self.v=nn.Linear(d_in,d_out)
#         self.dropout=nn.Dropout(dropout_prob)
#         self.linear=nn.Linear(d_out,d_out)
#         self.register_buffer("mask",torch.triu(torch.ones(self.sequence_length,self.sequence_length),diagonal=1))
#     def forward(self,x):
#         batch_size,sequence_length,d_in=x.shape
#         q = self.q(x).contiguous().view(batch_size,self.num_head,sequence_length,self.d_head)
#         k = self.k(x).contiguous().view(batch_size,self.num_head,sequence_length,self.d_head)
#         v = self.v(x).contiguous().view(batch_size,self.num_head,sequence_length,self.d_head)
#         attention_score=q@k.transpose(-1,-2)
#         attention_score.masked_fill_(self.mask.bool()[:sequence_length,:sequence_length],-torch.inf)
#         attention_score_softmax=torch.softmax(attention_score/math.sqrt(self.d_out),dim=-1)
#         attention_score_dropout=self.dropout(attention_score_softmax)
#         attention_score=attention_score_dropout@v
#         attention_score=attention_score.contiguous().view(batch_size,sequence_length,self.d_out)
#         attention_score=self.linear(attention_score)
#         return attention_score

# model=attentionblock(d_in=256,d_out=128,sequence_length=8,dropout_prob=0.3,d_head=128,num_head=2)
# print(model(input_embeddings).shape)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x)
        x = x + shortcut  

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + shortcut
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# id1=torch.tensor(tokenizer.encode(txt1))
# id2=torch.tensor(tokenizer.encode(txt2))
# input_x=torch.stack((id1,id2),dim=0)
# model=GPTModel(GPT_CONFIG_124M)
# print(model(input_x).shape)
# para_sum=sum(tensor.numel() for tensor in model.parameters())
# print(para_sum)
#计算parameter数量

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

#
# start_context = "Every effort moves you"
# tokenizer = tiktoken.get_encoding("gpt2")
#
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(start_context, tokenizer),
#     max_new_tokens=10,
#     context_size=GPT_CONFIG_124M["context_length"]
# )

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

def calc_loss_batch(input_batch, target_batch, model):
    input_batch, target_batch = input_batch, target_batch
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss
def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    print(i)
    return total_loss / num_batches
# calc_loss_loader(data_loader=train_loader,model=model, num_batches=None)

def evaluate_model(model, train_loader, val_loader,  eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model,  num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model,  num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate(model, idx, max_new_tokens, context_size,
             temperature=1.0, top_k=None, eos_id=None):
    for i in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,torch.tensor(float('-inf')).to(logits.device),logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate(model, tokenizer, start_context)
    return train_losses, val_losses, track_tokens_seen
# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=1,
#     start_context="Every effort moves you", tokenizer=tokenizer
# )

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

for i in range(5):
    input_text=input("请输入:")
    load_weights_into_gpt(gpt, params)
    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=50,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
torch.save(gpt.state_dict(), "/Users/apple/Desktop/Python/AI learning/MY_GPT/parameter data.py")



