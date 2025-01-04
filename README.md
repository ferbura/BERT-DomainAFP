# BERT-DomainAFP
### BERT-DomainAFP in Pytorch.

### Install
```
pip3 install protein-bert-pytorch 
pip3 install torch torchvision torchaudio
pip3 install biopython pandas numpy
 ```
###  Usage
```
python3 predict_protein.py -f example.fasta --model_path path_to_your_model.pt
```
## model building
#### A small batch random oversampling strategy is used in multiple model training, so the models trained in different batches differ in the fitting process.
### Install
```
pip3 install protein-bert-pytorch 
pip3 install torch torchvision torchaudio
pip3 install biopython pandas numpy scikit-learn tensorboard
 ```
### Change train_file_path, validation_file_path, test_file_path to your data paths.The training data for this study is stored in dataset&code/base_data.
```
#!/usr/bin/env python
# coding: utf-8

import torch
from protein_bert_pytorch import ProteinBERT, PretrainingWrapper
import pandas as pd
import ast
from torch.utils.data import random_split, DataLoader, TensorDataset 
from torch.utils.tensorboard import SummaryWriter
import os, shutil
import numpy as np  
from sklearn.utils import resample

# Set dataset file paths
train_file_path = "AntiFreezeDomains-Train.csv"
validation_file_path = "AntiFreezeDomains-Validation.csv"
test_file_path = "AntiFreezeDomains-Test.csv"

# Load the datasets
train_df = pd.read_csv(train_file_path)
validation_df = pd.read_csv(validation_file_path)
test_df = pd.read_csv(test_file_path)

model = ProteinBERT(
    num_tokens = 21,
    num_annotation = 27,
    dim = 1024,
    dim_global = 256,
    depth = 4,
    narrow_conv_kernel = 9,
    wide_conv_kernel = 9,
    wide_conv_dilation = 5,
    attn_heads = 6,
    attn_dim_head = 36,
    local_to_global_attn = False,
    local_self_attn = True,
    num_global_tokens = 2,
    glu_conv = False
)

learner = PretrainingWrapper(
    model,
    random_replace_token_prob = 0.05,  
    remove_annotation_prob = 0.25,       
    add_annotation_prob = 0.01,          
    remove_all_annotations_prob = 0.5,   
    seq_loss_weight = 1.,                
    annotation_loss_weight = 1.,         
    exclude_token_ids = ()        
)

class SequenceDataProcessor:  
    def __init__(self, train_df):  
        self.train_df = train_df  
        self.max_length = 1024
        self.vector_size = 27  
        self.code_dict = 'ACDEFGHIKLMNPQRSTVWY'  
        self.process_data()  

    def process_data(self):  
        self.train_df["Domain"] = self.train_df["Domain"].apply(ast.literal_eval)  
        self.train_df['Length'] = self.train_df['Sequence'].apply(len)  
        self.train_df = self.train_df[self.train_df['Length'] <= self.max_length]  
        self.train_df = self.train_df.drop(columns='Length')  
        self.train_df["Domain"] = self.train_df["Domain"].apply(self.create_encoding_vector)  
        self.seqs, self.masks, self.annotations = self.encode_sequences()  

    def create_encoding_vector(self, positions):  
        encoding_vector = [0] * self.vector_size  
        for pos in positions:  
            if 0 <= pos <= self.vector_size:  
                encoding_vector[pos-1] = 1  
        return encoding_vector  

    def one_hot_encode(self, sequence):  
        encoding = np.zeros(self.max_length)  
        mask = np.zeros(self.max_length)  
        for i, aa in enumerate(sequence):  
            if aa in self.code_dict and i < self.max_length:  
                encoding[i] = self.code_dict.index(aa)  
                mask[i] = 1  
        return encoding, mask  

    def encode_sequences(self):  
        seqs = []  
        masks = []  
        annotations = []  
        for sequence, domain in zip(self.train_df["Sequence"], self.train_df["Domain"]):  
            seq_encoding, seq_mask = self.one_hot_encode(sequence)  
            seqs.append(seq_encoding)  
            masks.append(seq_mask)  
            annotations.append(domain)  
        return np.array(seqs), np.array(masks), np.array(annotations)  

    def convert_to_torch_tensors(self):  
        self.seqs = torch.from_numpy(np.array([np.array(i).astype(int) for i in self.seqs])).long()  
        self.masks = torch.from_numpy(np.array([np.array(i).astype(bool) for i in self.masks]))  
        self.annotations = torch.from_numpy(np.array([np.array(i).astype(float) for i in self.annotations])).float()  

    def get_data(self):  
        return self.seqs, self.masks, self.annotations

def check_positions(vector, positions):  
    for position in positions:  
        if position < 1 or position > len(vector):  
            print(f"Position {position} is out of range")
            return [0, 0]  
        elif vector[position - 1] == 1:  
            return [position, 1]  
    return [0, 0]

def test_model(model, test_loader):  
    total_predict = np.array([])  
    label = np.array([])  
    for batch_seq, batch_annotation, batch_mask in test_loader:  
        predict = model(batch_seq, batch_annotation, mask=batch_mask)  
        binary_predictions = (predict[1] > 0).int()  
          
        if total_predict.size == 0:  
            total_predict = binary_predictions.numpy()  
            label = batch_annotation.numpy()  
        else:  
            total_predict = np.concatenate((total_predict, binary_predictions.numpy()), axis=0)  
            label = np.concatenate((label, batch_annotation.numpy()), axis=0)  

    right = 0  
    for i in range(len(total_predict)):  
        if check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])[1] == check_positions(label[i], [1, 2, 3, 4, 5, 27])[1]:  
            right += 1  
    AFP_predict = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 27: 0}  
    AFP_total = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 27: 0}  
    for i in range(len(total_predict)):  
        p = check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])  
        l = check_positions(label[i], [1, 2, 3, 4, 5, 27])  
        if p[0] == l[0]:  
            AFP_predict[l[0]] += 1
            AFP_total[l[0]] += 1
        else:
            AFP_total[l[0]] += 1
    accuracy = right / len(total_predict)  
    print(f"Model Accuracy: {accuracy * 100:.2f}%")  
    print(f"Classification Accuracy: {AFP_total, AFP_predict}")
    return accuracy

def oversample_to_balance(df, target_count=150, target_column='Type'):  
    oversampled_df = pd.DataFrame()  
    unique_types = df[target_column].unique()  
    for type_value in unique_types:  
        type_df = df[df[target_column] == type_value]  
        if len(type_df) < target_count:  
            type_df_upsampled = resample(type_df, replace=True, n_samples=target_count)  
            oversampled_df = pd.concat([oversampled_df, type_df_upsampled], ignore_index=True) 
        else:  
            type_df_upsampled = resample(type_df, replace=False, n_samples=target_count)  
            oversampled_df = pd.concat([oversampled_df, type_df_upsampled], ignore_index=True)  
    return oversampled_df  

def load_datasets_and_loaders(train_df, batch_size=32):  
    AFP_df = train_df[~train_df["Type"].isnull()]
    fake_AFP_df = train_df[train_df["Type"].isnull()]
    AFP_df = oversample_to_balance(AFP_df)
    fake_AFP_df = fake_AFP_df.sample(n=len(AFP_df))
    train_df = pd.concat([AFP_df, fake_AFP_df], ignore_index=True)
    sequence_rows = train_df[train_df['NAME'].str.startswith('Sequence_')]  
    num_fake_sequences = len(sequence_rows)  
    fake_sequence_rows = train_df[train_df['NAME'].str.startswith('fake_Sequence_')] 
    selected_fake_sequences = fake_sequence_rows.sample(n=num_fake_sequences, replace=False) 
    train_df = pd.concat([sequence_rows, selected_fake_sequences]) 
    processor = SequenceDataProcessor(train_df)  
    processor.convert_to_torch_tensors()  
    seqs, masks, annotations = processor.get_data()  
    train_dataset = TensorDataset(seqs, annotations, masks)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
    return train_loader

# Process train, validation, and test datasets
processor_train = SequenceDataProcessor(train_df)
processor_train.convert_to_torch_tensors()
seqs_train, masks_train, annotations_train = processor_train.get_data()
train_dataset = TensorDataset(seqs_train, annotations_train, masks_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

processor_validation = SequenceDataProcessor(validation_df)
processor_validation.convert_to_torch_tensors()
seqs_validation, masks_validation, annotations_validation = processor_validation.get_data()
validation_dataset = TensorDataset(seqs_validation, annotations_validation, masks_validation) 
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

processor_test = SequenceDataProcessor(test_df)
processor_test.convert_to_torch_tensors()
seqs_test, masks_test, annotations_test = processor_test.get_data()
test_dataset = TensorDataset(seqs_test, annotations_test, masks_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  
num_epochs = 12
best_acc = 0

for epoch in range(num_epochs):  
    print(f"Epoch: {epoch} Start")
    model.train()  
    total_loss = 0.0  
    batch_idx = 0  
    for batch_seq, batch_annotation, batch_mask in train_loader:  
        loss = learner(batch_seq, batch_annotation, mask=batch_mask)  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  
        total_loss += loss.item()  

    model.eval()  
    val_loss = 0.0  
    val_accuracy = 0.0  
    total_predict = np.array([])  
    label = np.array([])  
    with torch.no_grad():  
        right = 0
        for val_batch_seq, val_batch_annotation, val_batch_mask in validation_loader:  
            val_outputs = model(val_batch_seq, val_batch_annotation, mask=val_batch_mask)  
            val_loss = learner(val_batch_seq, val_batch_annotation, mask=val_batch_mask)    
            binary_predictions = (val_outputs[1] > 0).int()  
            
            if total_predict.size == 0:  
                total_predict = binary_predictions.numpy()  
                label = val_batch_annotation.numpy()   
            else:  
                total_predict = np.concatenate((total_predict, binary_predictions.numpy()), axis=0)  
                label = np.concatenate((label, val_batch_annotation.numpy()), axis=0)  

    for i in range(len(total_predict)):
        if check_positions(total_predict[i], [1, 2, 3, 4, 5, 27]) == check_positions(label[i], [1, 2, 3, 4, 5, 27]):
            right += 1  
    val_loss /= len(total_predict)  
    right /= len(total_predict)
    if right > best_acc:
        torch.save(model.state_dict(), './BERT-DomainAFP.pt')
        best_acc = right
    print(f"Loss: {val_loss}, Accuracy: {right}")

accuracy = test_model(model, test_loader)

```
# Develop a BERT model for one-way prediction from "sequence to structural domain".
```
import torch
import pandas as pd
import ast
from torch.utils.data import random_split, DataLoader, TensorDataset 
from torch.utils.tensorboard import SummaryWriter
import os,shutil
import pandas as pd  
import numpy as np  
import torch  
import ast  
from sklearn.utils import resample
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat


# helpers
train_data_path = "data/AntiFreezeDomains-Train.csv"
test_data_path = "data/AntiFreezeDomains-Test.csv"
validation_data_path = "data/AntiFreezeDomains-Validation.csv"




def exists(val):
    return val is not None

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class GlobalLinearSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head,
        heads
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, feats, mask = None):
        h = self.heads
        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n ()')
            k = k.masked_fill(~mask, -torch.finfo(k.dtype).max)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        if exists(mask):
            v = v.masked_fill(~mask, 0.)

        context = einsum('b h n d, b h n e -> b h d e', k, v)
        out = einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_keys,
        dim_out,
        heads,
        dim_head = 64,
        qk_activation = nn.Tanh()
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.qk_activation = qk_activation

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_keys, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim_out)

        self.null_key = nn.Parameter(torch.randn(dim_head))
        self.null_value = nn.Parameter(torch.randn(dim_head))

    def forward(self, x, context, mask = None, context_mask = None):
        b, h, device = x.shape[0], self.heads, x.device

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        null_k, null_v = map(lambda t: repeat(t, 'd -> b h () d', b = b, h = h), (self.null_key, self.null_value))
        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

        q, k = map(lambda t: self.qk_activation(t), (q, k))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask) or exists(context_mask):
            i, j = sim.shape[-2:]

            if not exists(mask):
                mask = torch.ones(b, i, dtype = torch.bool, device = device)

            if exists(context_mask):
                context_mask = F.pad(context_mask, (1, 0), value = True)
            else:
                context_mask = torch.ones(b, j, dtype = torch.bool, device = device)

            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            sim.masked_fill_(~mask, max_neg_value(sim))

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Layer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_global,
        narrow_conv_kernel = 9,
        wide_conv_kernel = 9,
        wide_conv_dilation = 5,
        attn_heads = 8,
        attn_dim_head = 64,
        attn_qk_activation = nn.Tanh(),
        local_to_global_attn = False,
        local_self_attn = False,
        glu_conv = False
    ):
        super().__init__()

        self.seq_self_attn = GlobalLinearSelfAttention(dim = dim, dim_head = attn_dim_head, heads = attn_heads) if local_self_attn else None

        conv_mult = 2 if glu_conv else 1

        self.narrow_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, narrow_conv_kernel, padding = narrow_conv_kernel // 2),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        self.wide_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, wide_conv_kernel, dilation = wide_conv_dilation, padding = wide_conv_padding),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        self.local_to_global_attn = local_to_global_attn

        if local_to_global_attn:
            self.extract_global_info = CrossAttention(
                dim = dim,
                dim_keys = dim_global,
                dim_out = dim,
                heads = attn_heads,
                dim_head = attn_dim_head
            )
        else:
            self.extract_global_info = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.Linear(dim_global, dim),
                nn.GELU(),
                Rearrange('b d -> b () d')
            )

        self.local_norm = nn.LayerNorm(dim)

        self.local_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
            )),
            nn.LayerNorm(dim)
        )

        self.global_attend_local = CrossAttention(dim = dim_global, dim_out = dim_global, dim_keys = dim, heads = attn_heads, dim_head = attn_dim_head, qk_activation = attn_qk_activation)

        self.global_dense = nn.Sequential(
            nn.Linear(dim_global, dim_global),
            nn.GELU()
        )

        self.global_norm = nn.LayerNorm(dim_global)

        self.global_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim_global, dim_global),
                nn.GELU()
            )),
            nn.LayerNorm(dim_global),
        )

    def forward(self, tokens, annotation, mask = None):
        if self.local_to_global_attn:
            global_info = self.extract_global_info(tokens, annotation, mask = mask)
        else:
            global_info = self.extract_global_info(annotation)

        # process local (protein sequence)

        global_linear_attn = self.seq_self_attn(tokens) if exists(self.seq_self_attn) else 0

        conv_input = rearrange(tokens, 'b n d -> b d n')

        if exists(mask):
            conv_input_mask = rearrange(mask, 'b n -> b () n')
            conv_input = conv_input.masked_fill(~conv_input_mask, 0.)

        narrow_out = self.narrow_conv(conv_input)
        narrow_out = rearrange(narrow_out, 'b d n -> b n d')
        wide_out = self.wide_conv(conv_input)
        wide_out = rearrange(wide_out, 'b d n -> b n d')

        tokens = tokens + narrow_out + wide_out + global_info + global_linear_attn
        tokens = self.local_norm(tokens)

        tokens = self.local_feedforward(tokens)

        # process global (annotations)

        annotation = self.global_attend_local(annotation, tokens, context_mask = mask)
        annotation = self.global_dense(annotation)
        annotation = self.global_norm(annotation)
        annotation = self.global_feedforward(annotation)

        return tokens, annotation

# main model

class ProteinBERT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens = 26,
        num_annotation = 8943,
        dim = 512,
        dim_global = 256,
        depth = 6,
        narrow_conv_kernel = 9,
        wide_conv_kernel = 9,
        wide_conv_dilation = 5,
        attn_heads = 8,
        attn_dim_head = 64,
        attn_qk_activation = nn.Tanh(),
        local_to_global_attn = False,
        local_self_attn = False,
        num_global_tokens = 1,
        glu_conv = False
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.num_global_tokens = num_global_tokens
        self.to_global_emb = nn.Linear(num_annotation, num_global_tokens * dim_global)

        self.layers = nn.ModuleList([Layer(dim = dim, dim_global = dim_global, narrow_conv_kernel = narrow_conv_kernel, wide_conv_dilation = wide_conv_dilation, wide_conv_kernel = wide_conv_kernel, attn_qk_activation = attn_qk_activation, local_to_global_attn = local_to_global_attn, local_self_attn = local_self_attn, glu_conv = glu_conv) for layer in range(depth)])

        self.to_token_logits = nn.Linear(dim, num_tokens)

        self.to_annotation_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim_global, num_annotation)
        )

    def forward(self, seq, annotation, mask = None):
        tokens = self.token_emb(seq)

        annotation = self.to_global_emb(annotation)
        annotation = rearrange(annotation, 'b (n d) -> b n d', n = self.num_global_tokens)

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation, mask = mask)

        tokens = self.to_token_logits(tokens)
        annotation = self.to_annotation_logits(annotation)
        return tokens, annotation

# pretraining wrapper

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

class PretrainingWrapper(nn.Module):
    def __init__(
        self,
        model,
        random_replace_token_prob = 0.05,
        remove_annotation_prob = 0.25,
        add_annotation_prob = 0.01,
        remove_all_annotations_prob = 0.5,
        seq_loss_weight = 1.,
        annotation_loss_weight = 1.,
        exclude_token_ids = (0, 1, 2)   # for excluding padding, start, and end tokens from being masked
    ):
        super().__init__()
        assert isinstance(model, ProteinBERT), 'model must be an instance of ProteinBERT'

        self.model = model

        self.random_replace_token_prob = random_replace_token_prob
        self.remove_annotation_prob = remove_annotation_prob
        self.add_annotation_prob = add_annotation_prob
        self.remove_all_annotations_prob = remove_all_annotations_prob

        self.seq_loss_weight = seq_loss_weight
        self.annotation_loss_weight = annotation_loss_weight

        self.exclude_token_ids = exclude_token_ids

    def forward(self, seq, annotation, mask = None):
        batch_size, device = seq.shape[0], seq.device

        seq_labels = seq
        annotation_labels = annotation

        if not exists(mask):
            mask = torch.ones_like(seq).bool()

        # prepare masks for noising sequence

        excluded_tokens_mask = mask

        for token_id in self.exclude_token_ids:
            excluded_tokens_mask = excluded_tokens_mask & (seq != token_id)

        random_replace_token_prob_mask = get_mask_subset_with_prob(excluded_tokens_mask, self.random_replace_token_prob)

        # prepare masks for noising annotation

        batch_mask = torch.ones(batch_size, device = device, dtype = torch.bool)
        batch_mask = rearrange(batch_mask, 'b -> b ()')
        remove_annotation_from_batch_mask = get_mask_subset_with_prob(batch_mask, self.remove_all_annotations_prob)

        annotation_mask = annotation > 0
        remove_annotation_prob_mask = get_mask_subset_with_prob(annotation_mask, self.remove_annotation_prob)
        add_annotation_prob_mask = get_mask_subset_with_prob(~annotation_mask, self.add_annotation_prob)
        remove_annotation_mask = remove_annotation_from_batch_mask & remove_annotation_prob_mask

        # generate random tokens

        random_tokens = torch.randint(0, self.model.num_tokens, seq.shape, device=seq.device)

        for token_id in self.exclude_token_ids:
            random_replace_token_prob_mask = random_replace_token_prob_mask & (random_tokens != token_id)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # noise sequence

        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, seq)

        # noise annotation

        noised_annotation = annotation + add_annotation_prob_mask.type(annotation.dtype)
        noised_annotation = noised_annotation * remove_annotation_mask.type(annotation.dtype)

        # denoise with model
        self.annotations1 = torch.zeros_like(torch.from_numpy(np.array([np.array(i).astype(float) for i in noised_annotation]))).float()
        seq_logits, annotation_logits = self.model(noised_seq, self.annotations1, mask = mask)

        # calculate loss

        seq_logits = seq_logits[mask]
        seq_labels = seq_labels[mask]

        seq_loss = F.cross_entropy(seq_logits, seq_labels, reduction = 'sum')
        annotation_loss = F.binary_cross_entropy_with_logits(annotation_logits, annotation_labels, reduction = 'sum')

        return seq_loss * self.seq_loss_weight + annotation_loss * self.annotation_loss_weight
test_df = pd.read_csv(test_data_path)
validation_df = pd.read_csv(validation_data_path)
train_df=train_data_path
print(len(validation_df),len(test_df))
model = ProteinBERT(
    num_tokens = 21,
    num_annotation = 27,
    dim = 1024,
    dim_global = 256,
    depth = 4,
    narrow_conv_kernel = 9,
    wide_conv_kernel = 9,
    wide_conv_dilation = 5,
    attn_heads = 6,
    attn_dim_head = 36,
    local_to_global_attn = False,
    local_self_attn = True,
    num_global_tokens = 2,
    glu_conv = False
)

learner = PretrainingWrapper(
    model,
    random_replace_token_prob = 0,  
    remove_annotation_prob = 0,   
    add_annotation_prob = 0.01,        
    remove_all_annotations_prob = 0.25,  
    seq_loss_weight = 0.,              
    annotation_loss_weight = 1.,     
    exclude_token_ids = ()    
)


# do the following in a loop for a lot of sequences and annotations


# In[3]:


class SequenceDataProcessor:  
    def __init__(self, train_df):  
        self.train_df = train_df  
        self.max_length =  1024
        self.vector_size = 27  
        self.code_dict = 'ACDEFGHIKLMNPQRSTVWY'  
        self.process_data()  
  
    def process_data(self):  
        self.train_df["Domain"] = self.train_df["Domain"].apply(ast.literal_eval)  
        self.train_df['Length'] = self.train_df['Sequence'].apply(len)  
        self.train_df = self.train_df[self.train_df['Length'] <= self.max_length]  
        self.train_df = self.train_df.drop(columns='Length')  
        self.train_df["Domain"] = self.train_df["Domain"].apply(self.create_encoding_vector)  
        self.seqs, self.masks, self.annotations = self.encode_sequences()  
  
    def create_encoding_vector(self, positions):  
        encoding_vector = [0] * self.vector_size  
        for pos in positions:  
            if 0 <= pos <= self.vector_size:  
                encoding_vector[pos-1] = 1  
        return encoding_vector  
  
    def one_hot_encode(self, sequence):  
        encoding = np.zeros(self.max_length)  
        mask = np.zeros(self.max_length)  
        for i, aa in enumerate(sequence):  
            if aa in self.code_dict and i < self.max_length:  
                encoding[i] = self.code_dict.index(aa)  
                mask[i] = 1  
        return encoding, mask  
  
    def encode_sequences(self):  
        seqs = []  
        masks = []  
        annotations = []  
        for sequence, domain in zip(self.train_df["Sequence"], self.train_df["Domain"]):  
            seq_encoding, seq_mask = self.one_hot_encode(sequence)  
            seqs.append(seq_encoding)  
            masks.append(seq_mask)  
            annotations.append(domain)  
        return np.array(seqs), np.array(masks), np.array(annotations)  
  
    def convert_to_torch_tensors(self):  
        self.seqs = torch.from_numpy(np.array([np.array(i).astype(int) for i in self.seqs])).long()  
        self.masks = torch.from_numpy(np.array([np.array(i).astype(bool) for i in self.masks]))  
        self.annotations = torch.from_numpy(np.array([np.array(i).astype(float) for i in self.annotations])).float()  
  
    def get_data(self):  
        return self.seqs, self.masks, self.annotations

def check_positions(vector, positions):  
    for position in positions:  
        if position < 1 or position > len(vector):  
            print(f"Position {position} is outside the range of the vector“")
            return [0,0]  
        elif vector[position - 1] == 1:
            return [position,1]  
    return [0,0]
    
def test_model(model, test_loader):  
    total_predict = np.array([])  
    label = np.array([])  
    for batch_seq, batch_annotation, batch_mask in test_loader:  
        batch_annotation1 = torch.zeros_like(torch.from_numpy(np.array([np.array(i).astype(float) for i in batch_annotation]))).float()
        predict = model(batch_seq, batch_annotation1, mask=batch_mask)  
        binary_predictions = (predict[1] > 0).int()  
          
        if total_predict.size == 0:  
            total_predict = binary_predictions.numpy()  
            label = batch_annotation.numpy()  
        else:  
            total_predict = np.concatenate((total_predict, binary_predictions.numpy()), axis=0)  
            label = np.concatenate((label, batch_annotation.numpy()), axis=0)  
          
        print(len(total_predict), len(label))  
    right = 0  
    for i in range(len(total_predict)):  
        if check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])[1] == check_positions(label[i], [1, 2, 3, 4, 5, 27])[1]:  
            right += 1  
    AFP_predict = {0:0,1:0,2:0,3:0,5:0,27:0}  
    AFP_total = {0:0,1:0,2:0,3:0,5:0,27:0}  
    for i in range(len(total_predict)):  
        p=check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])
        l=check_positions(label[i], [1, 2, 3, 4, 5, 27])
        if p[0] == l[0]:  
            AFP_predict[l[0]] += 1
            AFP_total[l[0]] += 1
        else:
            AFP_total[l[0]] += 1
    accuracy = right / len(total_predict)  
    print(f"accuracy: {accuracy * 100:.2f}%")  
    print(f"classification accuracy: {AFP_total,AFP_predict}")
    return accuracy

def oversample_to_balance(df, target_count=150, target_column='Type'):  
    oversampled_df = pd.DataFrame()  
    unique_types = df[target_column].unique()  
    for type_value in unique_types:  
        type_df = df[df[target_column] == type_value]  
        if len(type_df) < target_count:  
            type_df_upsampled = resample(type_df,   
                                          replace=True, 
                                          n_samples=target_count,
                                          #random_state=123
                                          ) 
            oversampled_df = pd.concat([oversampled_df,type_df_upsampled], ignore_index=True)  
        else:  
            type_df_upsampled = resample(type_df,   
                                          replace=False, 
                                          n_samples=target_count,
                                          #random_state=123
                                          )  
            oversampled_df = pd.concat([oversampled_df,type_df_upsampled], ignore_index=True)  
    return oversampled_df  





processor_test=SequenceDataProcessor(test_df)
processor_test.convert_to_torch_tensors()
seqs_test, masks_test, annotations_test = processor_test.get_data()
test_dataset = TensorDataset(seqs_test, annotations_test, masks_test) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)





processor_validation=SequenceDataProcessor(validation_df)
processor_validation.convert_to_torch_tensors()
seqs_validation, masks_validation, annotations_validation = processor_validation.get_data()
validation_dataset = TensorDataset(seqs_validation, annotations_validation, masks_validation) 
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

def load_datasets_and_loaders(train_file_path, batch_size=32):  
    # Read training and testing CSV files  
    train_df = pd.read_csv(train_file_path)  
    AFP_df=train_df[~train_df["Type"].isnull()]
    fake_AFP_df=train_df[train_df["Type"].isnull()]
    AFP_df=oversample_to_balance(AFP_df)
    #print(len(AFP_df))
    fake_AFP_df = fake_AFP_df.sample(n=int(len(AFP_df)/5))#int(len(AFP_df)/5)
    train_df=pd.concat([AFP_df,fake_AFP_df], ignore_index=True)
    #print(train_df)
    sequence_rows = train_df[train_df['NAME'].str.startswith('Sequence_')]  
    num_fake_sequences = int(len(AFP_df)/5) 
    fake_sequence_rows = train_df[train_df['NAME'].str.startswith('fake_Sequence_')] 
    selected_fake_sequences = fake_sequence_rows.sample(n=num_fake_sequences, replace=False) 
    train_df = pd.concat([sequence_rows, selected_fake_sequences]) 
    processor = SequenceDataProcessor(train_df)  
    #processor.imbalance_data()
    processor.convert_to_torch_tensors()  
    seqs, masks, annotations = processor.get_data()  
    train_dataset = TensorDataset(seqs, annotations, masks)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
    return train_loader


train_loader = load_datasets_and_loaders(train_df)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  
num_epochs = 50
best_acc=0
for epoch in range(num_epochs):  
    print(f"epoch:{epoch} start")
    model.train()  
    total_loss = 0.0  
    batch_idx = 0 
    for batch_seq, batch_annotation, batch_mask in train_loader:   
        loss = learner(batch_seq, batch_annotation, mask=batch_mask)  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  
        total_loss += loss.item()  
    train_loader=load_datasets_and_loaders(train_df)
    model.train()

    model.eval()   
    val_loss = 0.0  
    val_accuracy = 0.0  
    total_predict = np.array([])  
    label = np.array([])  
    with torch.no_grad(): 
        right=0
        for val_batch_seq, val_batch_annotation, val_batch_mask in validation_loader: 
            val_batch_annotation1 = torch.zeros_like(torch.from_numpy(np.array([np.array(i).astype(float) for i in val_batch_annotation]))).float()
            #_batch_annotation = torch.from_numpy(np.array([np.array(i).astype(float) for i in val_batch_annotation])).float()
            val_outputs = model(val_batch_seq, val_batch_annotation1, mask=val_batch_mask)  
            val_loss = learner(val_batch_seq,val_batch_annotation, mask=val_batch_mask)    
            binary_predictions = (val_outputs[1] > 0).int()  
            
            if total_predict.size == 0:  
                total_predict = binary_predictions.numpy()  
                label = val_batch_annotation.numpy()   
            else:  
                total_predict = np.concatenate((total_predict, binary_predictions.numpy()), axis=0)  
                label = np.concatenate((label, val_batch_annotation.numpy()), axis=0)  
        
        for i in range(len(total_predict)):
            if check_positions(total_predict[i], [1, 2, 3,  5, 27]) == check_positions(label[i], [1, 2, 3, 5, 27]):
                right += 1  
    print(len(total_predict),right)
    val_loss /= len(total_predict)  
    right /= len(total_predict)
    
    if right > best_acc:
        torch.save(model.state_dict(), f'./BERT-DomainAFP.pt')
        best_acc=right
        print(f"epoch:{epoch},loss{val_loss},acc{right}")
    accuracy = test_model(model, test_loader)
```

                                                                                                                     









