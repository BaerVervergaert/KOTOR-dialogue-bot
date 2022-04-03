#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from math import ceil, floor

import numpy as np
import pandas as pd

import os
import urllib
import ast


# In[2]:


### Config settings

## Data config
# Download the data again regardless if it already exists?
download_again = False


## DataLoader config
# Batch size
batch_size = 64


## Model config
# filenames for model storage
encoder_name = 'kotor-rnn-encoder-encinput.ptm'
decoder_name = 'kotor-rnn-decoder-encinput.ptm'


# In[3]:


use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
torch.backends.cudnn.benchmark = True


# In[4]:


url = 'https://github.com/hmi-utwente/video-game-text-corpora/raw/master/Star%20Wars:%20Knights%20of%20the%20Old%20Republic/data/dataset_20200716.csv'
filename = 'dataset_20200716.csv'
if (not os.path.exists(filename)) or download_again:
    urllib.request.urlretrieve(url,filename)


# In[5]:


index_col = 'id'
usecols = None
#usecols = ['id','text','previous']
converters = {'previous':ast.literal_eval,
              'next':ast.literal_eval,
             }
data = pd.read_csv(filename,
                   index_col=index_col,
                   usecols=usecols,
                   converters=converters,
                  )
data


# In[6]:


data.dtypes


# In[7]:


data.info()


# In[8]:


for col in data.columns:
    try:
        desc = data[col].apply(len).describe()
        print('\t',col)
        print(desc)
        print()
    except:
        pass


# In[9]:


def add_start_stop_codon(data,column,start='\r',stop='\n',force=False):
    detect_start_stop = lambda s,start=start,stop=stop: start in s or stop in s
    codons_in_text = data[column].apply(detect_start_stop).any()
    if codons_in_text:
        if not force:
            raise ValueError('data already contains start or stop codon at column: {0}'.format(column))
    transform = lambda s,start=start,stop=stop: start+s+stop
    data[column] = data[column].apply(transform)


# In[10]:


add_start_stop_codon(data,'text')
data['text']


# In[11]:


class CustomDataset:
    def __init__(self,data):
        self.data = data
        self.tensors = {}
        
        #Make Vocab
        self.vocab = sorted(list(set(''.join(self.data['text']))))
        self.ch2i = { v:k for k,v in enumerate(self.vocab) }
        self.i2ch = { v:k for k,v in self.ch2i.items() }
        
    def __len__(self):
        return(len(self.data))
    def str2vec(self,text):
        out = torch.tensor([ self.ch2i[s] for s in text],dtype=torch.long)
        return(out)
    def vec2str(self,vec):
        out = ''.join([ self.i2ch[i.item()] for i in vec ])
        return(out)
    def get_dialogue(self,idx):
        try:
            dialogue = self.tensors[idx]
        except:
            if idx is None:
                dialogue = '\r\n'
            else:
                dialogue = self.data.loc[idx,'text']
            dialogue = self.str2vec(dialogue)
            self.tensors[idx] = dialogue
        return(dialogue)
    def __getitem__(self,idx):
        self.data.loc[idx,'text']
        response = self.get_dialogue(idx)
        ins = response[:-1]
        outs = response[1:]
        prevs = self.data.loc[idx,'previous']
        prevs = np.random.choice(prevs)
        prevs = self.get_dialogue(prevs)
        return(prevs,ins,outs)


# In[12]:


dataset = CustomDataset(data)
print(len(dataset))
print()
print(dataset.vocab)
print()
print(dataset[0])
print()
print(dataset[len(dataset)-1])


# In[13]:


def collate_fn(data_points):
    L_prevs = [len(p) for p,i,o in data_points]
    L_currents  = [len(i) for p,i,o in data_points]
    N_prevs = max(*L_prevs)
    N_currents = max(*L_currents)
    B = len(data_points)
    prevs = torch.zeros((B,N_prevs),dtype=data_points[0][0].dtype)
    ins = torch.zeros((B,N_currents),dtype=data_points[0][1].dtype)
    outs = torch.zeros((B,N_currents),dtype=data_points[0][2].dtype)
    for k in range(B):
        l_prevs = L_prevs[k]
        prevs[k,:l_prevs] = data_points[k][0]
        l_currents = L_currents[k]
        ins[k,:l_currents] = data_points[k][1]
        outs[k,:l_currents] = data_points[k][2]
    return((prevs,ins,outs),L_prevs,L_currents)


# In[14]:


dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         pin_memory=True,
                                         collate_fn=collate_fn,
                                        )


# In[15]:


class TrainablePositionalEncoding(nn.Module):
    def __init__(self,encoding_dim,num_of_features=None):
        super(TrainablePositionalEncoding,self).__init__()
        self.num_of_features = num_of_features
        self.encoding_dim = encoding_dim
        if encoding_dim%2!=0:
            raise ValueError('encoding_dim should be a multiple of two!')
        if num_of_features is None:
            self.exp_linear = nn.Linear(1,encoding_dim//2,bias=True)
            self.angle_linear = nn.Linear(1,encoding_dim//2,bias=False)
        else:
            self.exp_linear = nn.Linear(self.num_of_features,encoding_dim//2,bias=True)
            self.angle_linear = nn.Linear(self.num_of_features,encoding_dim//2,bias=True)
    def forward(self,x):
        if self.num_of_features is None:
            x = x.unsqueeze(-1)
        exp_tensor = torch.exp(self.exp_linear(x)/80)
        angle_tensor = self.angle_linear(x)
        out = torch.cat((exp_tensor*torch.sin(angle_tensor),exp_tensor*torch.cos(angle_tensor)),dim=-1)
        return(out)


# In[16]:


def add_positional_info(x,ch2i = dataset.ch2i):
    try:
        space_idx = add_positional_info.space_idx
        punct_idx = add_positional_info.punct_idx
    except AttributeError:
        space_idx = ch2i[' ']
        punct_idx = torch.tensor([ ch2i[s] for s in ['.','!','?'] ]).to(device)
        add_positional_info.space_idx = space_idx
        add_positional_info.punct_idx = punct_idx
    punct_mask = torch.isin(x,punct_idx)
    punct_mask = punct_mask.cumsum(axis=1)
    space_mask = x==space_idx
    out = 0
    try:
        punct_mask_max = punct_mask.max().item()
    except RuntimeError:
        punct_mask_max = 0
    for punct_mark in range(punct_mask_max+1):
        punct_mark_mask = (punct_mask==punct_mark)
        out += (space_mask*punct_mark_mask).cumsum(axis=1)*punct_mark_mask
    out = torch.stack((x,out),dim=2)
    return(out)


# In[17]:


class Encoder(nn.Module):
    def __init__(self, vocab_size, out_dim, embedding_dim, rnn_units, n_layers=2):
        super(Encoder,self).__init__()
        self.n_layers = n_layers
        self.rnn_units = rnn_units
        self.out_dim = out_dim
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                     )
        self.pos_encoding = TrainablePositionalEncoding(embedding_dim)
        self.grus = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.initial_states = nn.ParameterList()
        for submodel_layers in range(1,n_layers+1):
            submodel_gru = nn.GRU(input_size=embedding_dim,
                                          hidden_size=self.rnn_units,
                                          num_layers=submodel_layers,
                                          batch_first=True,
                                         )
            submodel_linear = nn.Linear(rnn_units,
                                                out_dim,
                                                bias=True,
                                               )
            self.grus.append(submodel_gru)
            self.linears.append(submodel_linear)
            self.initial_states.append(nn.Parameter(torch.randn((submodel_layers,self.rnn_units,))))
        self.bias = nn.Parameter(torch.randn((out_dim,)))
    def batch_initial_states(self,batch_size):
        states = [ init_state.repeat((batch_size,1,1)).permute(1,0,2) for init_state in self.initial_states ]
        return(states)
    def forward(self, inputs, lengths, states=None,device=device):
        batch_size = len(lengths)
        if states is None:
            states = self.batch_initial_states(batch_size)
        x = self.embedding(inputs[...,0])
        x += self.pos_encoding(inputs[...,1].float())
        x = torch.nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        out = 0
        for k in range(len(self.grus)):
            # Apply GRU
            state = states[k]
            submodel_gru = self.grus[k]
            sub_out, state = submodel_gru(x,state)

            # Apply linear transform
            sub_out,_ = torch.nn.utils.rnn.pad_packed_sequence(sub_out, batch_first=True)
            sub_out = sub_out[torch.arange(len(lengths)),torch.tensor(lengths).to(device)-1]
            submodel_linear = self.linears[k]
            sub_out = submodel_linear(sub_out)
            
            # Collect in output
            out += sub_out
        return(out)
    def noisify(self,scale):
        with torch.no_grad():
            for p in self.grus.parameters():
                p.add_(torch.randn_like(p),alpha=scale)
            for p in self.linears.parameters():
                p.add_(torch.randn_like(p),alpha=scale)


# In[18]:


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, n_layers=2):
        super(Decoder,self).__init__()
        self.n_layers = n_layers
        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                     )
        self.pos_encoding = TrainablePositionalEncoding(embedding_dim)
        self.grus = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.initial_states = nn.ParameterList()
        for submodel_layers in range(1,n_layers+1):
            submodel_gru = nn.GRU(input_size=embedding_dim,
                                          hidden_size=self.rnn_units,
                                          num_layers=submodel_layers,
                                          batch_first=True,
                                         )
            submodel_linear = nn.Linear(rnn_units,
                                                vocab_size,
                                                bias=False,
                                               )
            self.grus.append(submodel_gru)
            self.linears.append(submodel_linear)
            self.initial_states.append(nn.Parameter(torch.randn((submodel_layers,self.rnn_units,))))
        self.bias = nn.Parameter(torch.randn((vocab_size,)))
    def batch_initial_states(self,batch_size):
        states = [ init_state.repeat((batch_size,1,1)).permute(1,0,2) for init_state in self.initial_states ]
        return(states)
    def forward(self, inputs, encoding_tensor, lengths, states=None):
        if states is None:
            states = self.batch_initial_states(len(lengths))
        batch_size = len(lengths)
        x = self.embedding(inputs[...,0])
        x += self.pos_encoding(inputs[...,1].float())
        x += encoding_tensor[:,None,:]
        x = torch.nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        out = 0
        for k in range(len(self.grus)):
            # Apply GRU
            state = states[k]
            submodel_gru = self.grus[k]
            sub_out, state = submodel_gru(x,state)
            states[k] = state
            
            # Apply linear transform
            sub_out,_ = torch.nn.utils.rnn.pad_packed_sequence(sub_out, batch_first=True)
            submodel_linear = self.linears[k]
            sub_out = submodel_linear(sub_out)
            
            # Collect in output
            out += sub_out
        
        out += self.bias[None,None,:]
        return(out,states)
    def noisify(self,scale):
        with torch.no_grad():
            for p in self.grus.parameters():
                p.add_(torch.randn_like(p),alpha=scale)
            for p in self.linears.parameters():
                p.add_(torch.randn_like(p),alpha=scale)


# In[19]:


vocab_size = len(dataset.vocab)
embedding_dim = 256
rnn_units = 256
n_layers = 5

encoder = Encoder(vocab_size=vocab_size,
                  out_dim = embedding_dim,
                  embedding_dim = embedding_dim,
                  rnn_units=rnn_units,
                  n_layers = n_layers,
             )
encoder.to(device)
print(encoder)

decoder = Decoder(vocab_size=vocab_size,
                  embedding_dim = embedding_dim,
                  rnn_units=rnn_units,
                  n_layers = n_layers,
             )
decoder.to(device)
print(decoder)


# In[20]:


print('encoder parameters:',sum( np.prod(p.shape) for p in encoder.parameters()))
print('decoder parameters:',sum( np.prod(p.shape) for p in decoder.parameters()))


# In[21]:


encoder.load_state_dict(torch.load(encoder_name))
encoder.to(device)
print('Loaded encoder')
decoder.load_state_dict(torch.load(decoder_name))
decoder.to(device)
print('Loaded decoder')


# In[22]:


prevs,ins,outs = dataset[0]
N_prevs = len(prevs)
N_currents = len(ins)
prevs = prevs.to(device)[None,:]
ins = ins.to(device)[None,:]
outs = outs.to(device)[None,:]
prevs = add_positional_info(prevs)
ins = add_positional_info(ins)
print(prevs.shape)
print(prevs)
print(dataset.vec2str(prevs[0,:,0]))
print(''.join( str(i) for i in prevs[0,:,1].tolist()[2:]))
print(ins.shape)
print(ins)
print(dataset.vec2str(ins[0,:,0]))
print(''.join( str(i) for i in ins[0,:,1].tolist()[2:]))
encoder_tensor = encoder(prevs,[N_prevs])
print(encoder_tensor.shape)
print(encoder_tensor)
pred, states = decoder(ins,encoder_tensor,[N_currents])
print(states)
print(dataset.vec2str(torch.exp(pred).squeeze(0).multinomial(1)))


# In[23]:


prevs,ins,outs = dataset[len(dataset)-1]
N_prevs = len(prevs)
N_currents = len(ins)
prevs = prevs.to(device)[None,:]
ins = ins.to(device)[None,:]
outs = outs.to(device)[None,:]
prevs = add_positional_info(prevs)
ins = add_positional_info(ins)
print(prevs.shape)
print(prevs)
print(dataset.vec2str(prevs[0,:,0]))
print(''.join( str(i) for i in prevs[0,:,1].tolist()[2:]))
print(ins.shape)
print(ins)
print(dataset.vec2str(ins[0,:,0]))
print(''.join( str(i) for i in ins[0,:,1].tolist()[2:]))
encoder_tensor = encoder(prevs,[N_prevs])
print(encoder_tensor.shape)
print(encoder_tensor)
pred, states = decoder(ins,encoder_tensor,[N_currents])
print(states)
print(dataset.vec2str(torch.exp(pred).squeeze(0).multinomial(1)))


# In[24]:


class Bot(nn.Module):
  def __init__(self, encoder, decoder, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    if type(temperature)==float or type(temperature)==int:
        temperature = lambda pred,*args,temp=temperature: pred/temp
    self.temperature = temperature
    self.encoder = encoder
    self.decoder = decoder
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars
    self.softmax = nn.Softmax(-1)

  def generate_answer(self, inputs, max_length=500):
    self.encoder.eval()
    self.decoder.eval()
    # Convert strings to token IDs.
    input_ids = self.ids_from_chars(inputs)
    input_ids = input_ids[None,:]
    input_ids = input_ids.to(device)
    input_ids = add_positional_info(input_ids)
    
    # Encode input
    encoder_tensor = self.encoder(input_ids,[len(inputs)])

    # First Run
    input_ids = self.ids_from_chars('\r')
    input_ids = input_ids[None,:]
    input_ids = input_ids.to(device)
    input_ids = add_positional_info(input_ids)
    last_positional_info = input_ids[:,-1:,1]
    predicted_logits, states = self.decoder(input_ids,encoder_tensor,[1])
    
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    prediction_position = len(inputs)
    predicted_logits = self.temperature(predicted_logits,prediction_position)

    # Sample the output logits to generate token IDs.
    predicted_logits = self.softmax(predicted_logits)
    predicted_ids = predicted_logits.multinomial(1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids[0])
    predicted_chars = list(predicted_chars)
    predicted_chars = predicted_chars[0]
    if predicted_chars == ' ':
        last_positional_info += 1
    elif predicted_chars in ['.','!','?']:
        last_positional_info *= 0

    run = '\r' + predicted_chars
    for _ in range(max_length):
        # Consecutive Run
        predicted_ids = torch.stack((predicted_ids,last_positional_info),dim=2)
        predicted_logits, states = self.decoder(predicted_ids,encoder_tensor,[1],states)

        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        prediction_position += 1
        predicted_logits = self.temperature(predicted_logits,prediction_position)

        # Sample the output logits to generate token IDs.
        predicted_logits = self.softmax(predicted_logits)
        predicted_ids = predicted_logits.multinomial(1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids[0])
        predicted_chars = list(predicted_chars)
        predicted_chars = predicted_chars[0]
        
        # Update positional info
        if predicted_chars == ' ':
            last_positional_info += 1
        elif predicted_chars in ['.','!','?']:
            last_positional_info *= 0

        run = run + predicted_chars
        if predicted_chars=='\n':
            break
    
    if run[-1]!='\n':
        run += '\n'
    
    return run


# In[25]:


bot = Bot(encoder,decoder,dataset.vec2str,dataset.str2vec)


# In[26]:


bot.generate_answer('\rWhat are you thinking?\n')


# In[27]:


#temperature = lambda pred,x,b=0: (pred-b)*(pred>b)+(pred-b)*(pred<b)/.33 + b
temperature = lambda pred,x,b=.9: pred/(b+(1.-b)/np.sqrt(.1*x+1))
bot = Bot(encoder,decoder,dataset.vec2str,dataset.str2vec,temperature=temperature)
bot.generate_answer('\rWhat are you thinking?\n')


# In[28]:


query = 'Got any money?'
print('Query:',query,end='\n\n\n')
for _ in range(10):
    print(bot.generate_answer('\r'+query+'\n'))


# In[29]:


while True:
    query = input()
    print('Query:',query,end='\n\n\n')
    print(bot.generate_answer('\r'+query+'\n'),end='\n\n')


# In[ ]:




