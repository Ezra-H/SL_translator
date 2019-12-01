#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
import pickle
import time
import pandas as pd

from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from my_model import sign_language_model,Hypothesis
import numpy as np
from tqdm import tqdm
from utils import *
from data import *

import torch
import torch.nn.utils
import torch.nn as nn
from torchvision import transforms
import argparse
from torch.utils.data import Dataset, Subset, DataLoader

parser = argparse.ArgumentParser(description='SLT')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--lr_decay',
    type=float,
    default=0.95,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--embedding_size',
    type=int,
    default=256,
    metavar='Es',
    help='embedding_size (default: 256)')
parser.add_argument(
    '--hidden_size',
    type=int,
    default=512,
    metavar='Hs',
    help='hidden_size (default: 512)')
parser.add_argument(
    '--dropout',
    type=float,
    default=0.3,
    metavar='Dp',
    help='parameter for dorpout (default: 0.5)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=8,
    metavar='W',
    help='how many training processes to use (default: 10)')
parser.add_argument(
    '--train_src',
    type=str,
    default='/data/shanyx/hrh/sign/ccsl/picture/',
    metavar='NS',
    help='the path of source picture (default: /data/shanyx/hrh/sign/ccsl/picture/)')
parser.add_argument(
    '--vail_src',
    type=str,
    default='/data/shanyx/hrh/sign/ccsl/picture/',
    metavar='NS',
    help='the path of source picture (default: /data/shanyx/hrh/sign/ccsl/picture/)')
parser.add_argument(
    '--vocab',
    default='/data/shanyx/hrh/sign/ccsl/vocab.csv',
    metavar='ENV',
    help='The vocab path (default: /data/shanyx/hrh/sign/ccsl/vocab.csv)')
parser.add_argument(
    '--load', default=True, metavar='L', help='load a trained model')
parser.add_argument(
    '--save_step',
    default=10,
    metavar='SS',
    help='Save model num of epoch')
parser.add_argument(
    '--gpus',
    type=int,
    default=[1],
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--clip_num',
    type=int,
    default=8,
    metavar='CN',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--uniform_init',
    type=float,
    default=0.1,
    metavar='CN',
    help='uniform_init (default: 0.1)')
parser.add_argument(
    '--beam_size',
    type=int,
    default=5,
    metavar='BS',
    help='beam-size (default: 5)')
parser.add_argument(
    '--clip_grad',
    type=int,
    default=5,
    metavar='CG',
    help='beam-size (default: 5)')
parser.add_argument(
    '--save_path',
    type=str,
    default='./checkpoints/model.pth',
    metavar='CG',
    help='the path to save the model (default:./checkpoints/model.pth')
parser.add_argument(
    '--patience',
    type=int,
    default=5,
    metavar='P',
    help='the patience to waiting the epoch alter (default:./checkpoints/model.pth')
parser.add_argument(
    '--max_num_trial',
    type=int,
    default=5,
    metavar='NT',
    help='(default: 5')
parser.add_argument(
    '--freeze',
    type=int,
    default=0,
    metavar='FZ',
    help='freeze the 3D conv(default: False')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    metavar='Bs',
    help='batch size(default: 1')
parser.add_argument(
    '--max_decoding_time_step',
    type=int,
    default=70,
    metavar='Bs',
    help='batch size(default: 70')
parser.add_argument(
    '--output_file',
    type=str,
    default='./checkpoints/predict.csv',
    metavar='OF',
    help='path to save the prediction, must be csv(default: ./checkpoints/predict.csv')




def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            if args.gpus[0] > -1:
                src_sents = src_sents.cuda()
            
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references, hypotheses):
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    
    # 图片补零准备
    _,c,n,h,w = batch[0][0].size()
    max_len = max([len(batch[b][0]) for b in range(batch_size)])
    pad = torch.zeros((max_len,c,n,h,w))

    # # 句子补零准备
    # se_max_len = max([len(batch[b][1]) for b in range(batch_size)])
    # se_pad = ['<\s>']*se_max_len

    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            temp = torch.zeros_like(pad)
            temp[:len(batch[b][0]),:,:,:,:] = batch[b][0]
            images.append(temp)  # 已经对图片进行对齐操作
            labels.append(batch[b][1])
    images = torch.stack(images, 0)
    return images, labels

def decode(args):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """
    root_dir = "/data/shanyx/hrh/sign/ccsl/picture/"
    csv_file = "/data/shanyx/hrh/sign/ccsl/corpus.csv"
    tf = transforms.Compose([
            transforms.Resize((260,210)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ])

    vocab = read_vocab(args.vocab)
    model = sign_language_model(embed_size=args.embedding_size,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout,
                vocab=vocab,
                args=args)
    print("load model from {}".format(args.save_path), file=sys.stderr)
    if args.gpus[0]>-1:
        model.load_state_dict(torch.load(args.save_path))
    else:
        model.load_state_dict(torch.load(args.save_path,map_location=torch.device('cpu')))
    
    if args.gpus[0] > -1:
        if len(args.gpus)>1:
            model = nn.DataParallel(model.cuda())
        else:
            model = model.cuda()

    train_dataset = ChineseSignDataset(root_dir,csv_file,num_frames_per_clip=8,transform=tf)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,   #现在只能load一个batch，但是一个batch中有一组图片，对应一个翻译
                            shuffle=True,
                            collate_fn=train_collate,
                            num_workers=args.workers)

    print("load test target sentences from [{}]".format(args.vocab), file=sys.stderr)
    test_data_tgt = train_dataset.words

    hypotheses = beam_search(model, train_loader,
                             beam_size=args.beam_size,
                             max_decoding_time_step=args.max_decoding_time_step)

    if test_data_tgt:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    # save hypotheses
    test_name = train_dataset.dirs
    df = pd.DataFrame(columns=['names','hypotheses'],data=zip(test_name,hypotheses))
    df.to_csv(args.output_file,index=False)


def beam_search(model, test_data_src, beam_size, max_decoding_time_step):
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_picts,__ in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            if model.args.gpus[0] > -1:
                src_picts = src_picts.cuda()
            
            example_hyps = model.beam_search(src_picts, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)
    return hypotheses


def main(args):
    """ Main func.
    """
    seed = args.seed
    torch.manual_seed(seed)
    if args.gpus[0] > -1:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    
    decode(args)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in args.gpus])
    main(args)
