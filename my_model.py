import os
import sys
from typing import List, Tuple, Dict, Set, Union
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from models import *
from data import *
from utils import *
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class sign_language_model(nn.Module):
    def __init__(self,args, vocab, embed_size=512, hidden_size=1024,enc_num_layers=2, dropout_rate=0.2):
        super(sign_language_model,self).__init__()
        self.args = args

        self.vocab = vocab
        self.feat_extraction = P3D19()
        if args.freeze:
            print('freeze 3D ResNet')
            for p in self.parameters():
                p.requires_grad=False
        self.encoder = nn.LSTM(embed_size, hidden_size,
                        num_layers=enc_num_layers,
                        dropout=dropout_rate,
                        bidirectional=True)
        self.lstm_decoder = nn.LSTMCell(embed_size+hidden_size, hidden_size,bias=True)
        self.ctc_decoder = None

        # 得到每个视频clip对应的词的概率
        self.fc1 = nn.Linear(2*hidden_size,len(vocab),bias=True)

        self.model_embeddings = nn.Embedding(len(vocab), embed_size)

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.enc_num_layers = enc_num_layers
        
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
        target_masks = torch.ones(target.shape).float()
        if self.args.gpus[0] > -1:
            target_masks = target_masks.cuda()
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        
        return scores


    def encode(self, source, source_lengths):
        # source is the images list

        # ## 直接使用view来进行拼接，通过特征提取器一次完成提取工作。再使用view来还原维度。  有问题，内存爆炸：contiguous
        # batch_size,seq_len,channel,num_clips,h,w = source.size()
        # source = source.reshape(-1,channel,num_clips,h,w).contiguous()
        # feats = self.feat_extraction(source)
        # feats = feats.view(batch_size,seq_len,-1)
        # feats = feats.permute(1,0,2) # 得到(len,b,embed_size)

        # 单独一个一个输入，得到的再进行拼接
        feats = []
        for i in source:
            feat = self.feat_extraction(i) # 返回 (len,embed_size)
            feats.append(feat) 
        feats = torch.stack(feats,1)  # 得到(len,b,embed_size)
        b = feats.size()[1]

        h0 = torch.randn(2*self.enc_num_layers,b, self.hidden_size)
        c0 = torch.randn(2*self.enc_num_layers,b, self.hidden_size)
        if self.args.gpus[0] > -1:
            h0 = h0.cuda()
            c0 = c0.cuda()
        
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
        target_id = torch.ones((len(target),max([len(i) for i in target]))).long()
        for i in range(len(target)):
            for j in range(len(target[i])):
                target_id[i][j] = self.vocab[target[i][j]]
        if self.args.gpus[0] > -1:
            target_id = target_id.cuda()
        return torch.t(target_id)
    
    def id2word(self,target):
        return list(self.vocab)[target]

    def beam_search(self, src_picts, beam_size=5, max_decoding_time_step=70):
        """ Given a single source video, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source video (pictures)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        # src_sents_var = self.vocab.to_input_tensor([src_picts], self.device)

        src_encodings, dec_init_vec = self.encode(src_picts, [len(src_picts)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size)
        if self.args.gpus[0] > -1:
            att_tm1 = att_tm1.cuda()

        eos_id = self.vocab['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float)
        if self.args.gpus[0] > -1:
            hyp_scores = hyp_scores.cuda()
        
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab[hyp[-1]] for hyp in hypotheses], dtype=torch.long)
            
            if self.args.gpus[0]>-1:
                y_tm1 = y_tm1.cuda()

            y_t_embed = self.model_embeddings(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            # log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)
            log_p_t = F.log_softmax(self.fc2(att_t), dim=-1)
            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.id2word(hyp_word_id)
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long)
            if self.args.gpus[0] > -1:
                live_hyp_ids = live_hyp_ids.cuda()
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float)
            if self.args.gpus[0] > -1:
                hyp_scores = hyp_scores.cuda()
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

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
