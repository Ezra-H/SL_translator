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
    def __init__(self,args, vocab, embed_size=512, hidden_size=1024,enc_num_layers=2, dropout_rate=0.2):
        super(sign_language_model,self).__init__()
        self.args = args

        self.vocab = vocab
        self.encoder = None
        self.decoder = None

        # 得到每个视频clip对应的词的概率
        self.fc1 = nn.Linear(2*hidden_size,len(vocab),bias=True)
        #self.fc2 = nn.Linear()

        self.model_embeddings = nn.Embedding(len(vocab), embed_size)

        #######
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.enc_num_layers = enc_num_layers
        ####################
        
        self.feat_extraction = P3D19()
        
        self.encoder = nn.LSTM(embed_size, hidden_size,
                        num_layers=enc_num_layers,
                        dropout=dropout_rate,
                        bidirectional=True)
        self.lstm_decoder = nn.LSTMCell(embed_size+hidden_size, hidden_size,bias=True)
        self.ctc_decoder = None

        self.h_projection = nn.Linear(2*enc_num_layers*hidden_size,hidden_size,bias=False)#(Linear Layer with no bias), called W_{h} in the PDF.
        self.c_projection = nn.Linear(2*enc_num_layers*hidden_size,hidden_size,bias=False)#(Linear Layer with no bias)# called W_{c} in the PDF.
        # default values
        self.att_projection = nn.Linear(2*hidden_size,hidden_size,bias=False)#(Linear Layer with no bias), called W_{attProj} in the PDF.
        self.combined_output_projection = nn.Linear(3*hidden_size,hidden_size,bias = False) #(Linear Layer with no bias), called W_{u} in the PDF.
        self.fc2 = nn.Linear(hidden_size,len(vocab),bias=False)#(Linear Layer with no bias), called W_{vocab} in the PDF.
        self.dropout = nn.Dropout(p=dropout_rate)

        ############ 

    def forward(self, source, target):
        """
        @param source (List[List[str]]): list of source images, just one bunch
        @param target (List[List[str]]): list of target sentence tokens of the source images, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        source_lengths = [len(s) for s in source] # this len of these images from one sentence
        enc_hiddens, dec_init_state = self.encode(source, source_lengths)
        
        Y_all=[]
        for i in enc_hiddens:
            Y_all.append(self.fc1(enc_hiddens))

        target = self.word2id(target)  # tokenize   is the transpose matrix
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target)
        
        P = F.log_softmax(self.fc2(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text  # the position of the pad is the zero value
        # target_masks = (target != self.vocab['<pad>']).float()
        target_masks = torch.ones(target.shape).float().cuda()
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        
        return scores


    def encode(self, source, source_lengths):
        # source is the images list
        feats = []
        for i in source:
            feat = self.feat_extraction(i) # 返回 (len,embed_size)
            feats.append(feat) 
        feats = torch.stack(feats,1)  # 得到(len,b,embed_size)
        b = feats.size()[1]

        h0 = torch.randn(2*self.enc_num_layers,b, self.hidden_size).cuda()
        c0 = torch.randn(2*self.enc_num_layers,b, self.hidden_size).cuda()
        
        enc_hiddens,(Last_hidden,Last_cell)  = self.encoder(feats,(h0,c0))
        enc_hiddens = enc_hiddens.permute(1,0,2)  # => (batch_size, seq_len, hidden_size)
        # print('enc_hiddens',enc_hiddens.size())  #dim=3 (image_num batch_size  hidden_size*2: bidirection)
        # print('Last_hiddens',Last_hidden.size())  #dim=2 (batch, hidden_size)

        init_decoder_hidden = self.h_projection(torch.cat((Last_hidden[0],Last_hidden[1], Last_hidden[2], Last_hidden[3]), dim=1))
        init_decoder_cell = self.c_projection(torch.cat((Last_cell[0], Last_cell[1], Last_cell[2], Last_cell[3]), dim=1))
        dec_init_state = (init_decoder_hidden , init_decoder_cell )
        
        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens, enc_masks, dec_init_state, target):
        # Chop of the <END> token for max length sentences.
        target = target[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size)

        if self.args.gpus[0] > -1:
            o_prev = o_prev.cuda()

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

    def step(self, Ybar_t,dec_state,enc_hiddens,enc_hiddens_proj, enc_masks) :
 
        combined_output = None
        dec_state = self.lstm_decoder(Ybar_t,dec_state)
        
        (dec_hidden,dec_cell) = dec_state
        
        e_t = torch.squeeze(torch.bmm(enc_hiddens_proj,torch.unsqueeze(dec_hidden,dim=2)),dim=2)
         
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        alpha_t = F.softmax(e_t,dim=1)
        
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t,dim=1),enc_hiddens),dim=1)
        
        U_t = torch.cat((dec_hidden , a_t),dim=1) #b,3h
        
        v_t = self.combined_output_projection(U_t)
        
        O_t = self.dropout(torch.tanh(v_t))

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens, source_lengths):
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        # 在真实位置填入中止符号
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)

        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        if self.args.gpus[0] > -1:
            enc_masks = enc_masks.cuda()
        return enc_masks

    def word2id(self, target):
        
        target_id = torch.zeros(target.shape).long()
        if self.args.gpus[0] > -1:
            target_id = target_id.cuda()
        for i in range(len(target)):
            for j in range(len(target[i])):
                target_id[i][j] = self.vocab[target[i][j]]
        return torch.t(target_id)

if __name__ == "__main__":
    torch.cuda.set_device(0)
    # vocab_path="F:/学习资料/基金项目-手语/nslt-master/Data/phoenix2014T.vocab.gloss"
    vocab_path = '/data/shanyx/hrh/sign/phoenix/phoenix2014T.vocab.gloss'
    vocab = read_vocab(vocab_path)
    model = sign_language_model(vocab)
    model = model.cuda()
    source=torch.autograd.Variable(torch.rand(2,3,16,260,210)).cuda()   # if modality=='Flow', please change the 2nd dimension 3==>2
    target = "test"
    #print(model)
    return_data = model(source,target)
    print(return_data[0].size(),return_data[1].size())
    pass
