import os
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from models import *
from data import *
from utils import *


class sign_language_model(nn.Module):
    def __init__(self,vocab, embed_size=256, hidden_size=256,enc_num_layers=2, dropout_rate=0.2):
        super(sign_language_model,self).__init__()
        self.encode_output_size = 2048  #layer=2
        self.vocab = vocab
        self.encoder = P3D19(embed_size = embed_size, hidden_size=hidden_size,enc_num_layers=enc_num_layers,bidirectional = True)
        self.decoder = None
        self.fc1 = nn.Linear(2*hidden_size,len(vocab),bias=True)
        #self.fc2 = nn.Linear()

        #######
        self.model_embeddings = model_embeddings(vocab, embed_size)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # default values
        #self.encoder = nn.LSTM(embed_size,hidden_size,bias=True,bidirectional = True)
        self.decoder = nn.LSTMCell(embed_size+hidden_size, hidden_size,bias=True)
        
        self.h_projection = nn.Linear(2*enc_num_layers*hidden_size,hidden_size,bias=False)#(Linear Layer with no bias), called W_{h} in the PDF.
        self.c_projection = nn.Linear(2*enc_num_layers*hidden_size,hidden_size,bias=False)#(Linear Layer with no bias)# called W_{c} in the PDF.

        self.att_projection = nn.Linear(2*enc_num_layers*hidden_size,hidden_size,bias=False)#(Linear Layer with no bias), called W_{attProj} in the PDF.
        self.combined_output_projection = nn.Linear(3*hidden_size,hidden_size,bias = False) #(Linear Layer with no bias), called W_{u} in the PDF.
        self.target_vocab_projection = nn.Linear(hidden_size,len(vocab),bias=False)#(Linear Layer with no bias), called W_{vocab} in the PDF.
        self.dropout = nn.Dropout(p=dropout_rate)

        ########

    def forward(self, source, target):
        """
        @param source (List[List[str]]): list of source images, just one bunch
        @param target (List[List[str]]): list of target sentence tokens of the source images, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        source_length = [len(s) for s in source] # this len of these images from one sentence
        self.encode(source,source_length)
        
        Y_all=[]
        for i in enc_hiddens:
            Y_all.append(self.fc1(enc_hiddens))


        #enc_hiddens, dec_init_state = self.encoder(source) #h,c  c是作为最后一个的隐向量


        
        #combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)

        #out,hn = self.decode(out,hn)
        
        return dec_init_state
        pass

    def encode(self, source, source_lengths):

        enc_hiddens, (Last_hidden, Last_cell) = self.encoder(source) #h,c  c是作为最后一个的隐向量
        #enc_hiddens = torch.squeeze(enc_hiddens,1)
        enc_hiddens = enc_hiddens.permute(1,0,2) #转换成batch_size在第一维度
        print(enc_hiddens.size())  #dim=3 (image_num batch_size  hidden_size*2: bidirection)
        print(Last_hidden.size())  #dim=2 (batch, hidden_size)
        init_decoder_hidden = self.h_projection(torch.cat((Last_hidden[0],Last_hidden[1], Last_hidden[2], Last_hidden[3]), dim=1))
        init_decoder_cell = self.c_projection(torch.cat((Last_cell[0], Last_cell[1], Last_cell[2], Last_cell[3]), dim=1))
        dec_init_state = (init_decoder_hidden , init_decoder_cell )
        
        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens, enc_masks, dec_init_state, target):
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []
        
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        
        Y = self.model_embeddings(target)
        
        for Y_t in torch.split(Y,1):
            Y_t = torch.squeeze(Y_t,dim=0)
            Ybar_t = torch.cat((Y_t,o_prev),dim=1)
            dec_state,o_t,e_t=self.step(Ybar_t,dec_state,enc_hiddens,enc_hiddens_proj,enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
        
        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs

def step(self, Ybar_t,dec_state,enc_hiddens,enc_hiddens_proj) :

        combined_output = None
        dec_state = self.decoder(Ybar_t,dec_state)
        
        (dec_hidden,dec_cell) = dec_state
        
        e_t = torch.squeeze(torch.bmm(enc_hiddens_proj,torch.unsqueeze(dec_hidden,dim=2)),dim=2)

        alpha_t = F.softmax(e_t,dim=1)
        
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t,dim=1),enc_hiddens),dim=1)
        
        U_t = torch.cat((dec_hidden , a_t),dim=1) #b,3h
        
        v_t = self.combined_output_projection(U_t)
        
        O_t = self.dropout(torch.tanh(v_t))

        combined_output = O_t
        return dec_state, combined_output, e_t

if __name__ == "__main__":
    torch.cuda.set_device(0)
    vocab_path="F:/学习资料/基金项目-手语/nslt-master/Data/phoenix2014T.vocab.gloss"
    vocab = read_vocab(vocab_path)
    model = sign_language_model(vocab)
    model = model.cuda()
    source=torch.autograd.Variable(torch.rand(2,3,16,260,210)).cuda()   # if modality=='Flow', please change the 2nd dimension 3==>2
    target = "test"
    #print(model)
    return_data = model(source,target)
    print(return_data[0].size(),return_data[1].size())

    pass
