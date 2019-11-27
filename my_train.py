#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
import pickle
import time


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from my_model import sign_language_model
import numpy as np
from tqdm import tqdm
from utils import *
from data import *

import torch
import torch.nn.utils
from torchvision import transforms
import argparse
from torch.utils.data import Dataset, Subset, DataLoader
# Subset(Dataset, [0,1,2,3,4])

parser = argparse.ArgumentParser(description='NMT')
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
    default=5,
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
    '--max_epoch',
    type=int,
    default=20000,
    metavar='M',
    help='maximum length of an episode (default: 20000)')
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
    default=[2],
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
    metavar='CG',
    help='the patience to waiting the epoch alter (default:./checkpoints/model.pth')
def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.append(batch[b][0])  # 对于视频的补长还待研究，主要是现在没有那么大的显存来载入这些图片。
            labels.append(batch[b][1])
    images = torch.stack(images, 0)
    labels = np.array(labels)
    return images, labels


def evaluate_ppl(args, model, dev_dataset):
    # waiting to correct

    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    vail_loader = DataLoader(dataset=dev_dataset,
                            batch_size=1,   #现在只能load一个batch，但是一个batch中有一组图片，对应一个翻译
                            shuffle=True,
                            collate_fn=train_collate,
                            num_workers=args.workers)

    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for imgs, target in tqdm(vail_loader):
            if args.gpus[0] > -1:
                imgs = imgs.cuda()

            loss = -model(imgs, target).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in target)  # omitting leading `<s>`
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


def train(args):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """

    root_dir = "/data/shanyx/hrh/sign/ccsl/picture/"
    csv_file = "/data/shanyx/hrh/sign/ccsl/corpus.csv"
    tf = transforms.Compose([
            transforms.Resize((260,210)),
            transforms.ToTensor()
            ])

    train_dataset = ChineseSignDataset(root_dir,csv_file,num_frames_per_clip=8,transform=tf)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=1,   #现在只能load一个batch，但是一个batch中有一组图片，对应一个翻译
                            shuffle=True,
                            collate_fn=train_collate,
                            num_workers=args.workers)
    
    clip_grad = args.clip_grad

    vocab = read_vocab(args.vocab)

    model = sign_language_model(embed_size=args.embedding_size,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout,
                vocab=vocab,
                args=args)
    model.train()
    

    uniform_init = args.uniform_init
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #load model
    if args.load:
        model.load_state_dict(torch.load(args.save_path))
        print('restore parameters of the optimizers', file=sys.stderr)
    if args.gpus[0] > -1:
        model = model.cuda()
    num_trial = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    for epoch in range(args.max_epoch):
        step = 0
        for imgs,label in tqdm(train_loader):
            # step: the index of data_loader
            # data: the return item of Class Dataset ,etc. (img_datas,img_label)
            step += 1
            if args.gpus[0] > -1:
                imgs = imgs.cuda()

            optimizer.zero_grad()

            example_losses = -model(imgs, label) # (batch_size) b-length value of the loss 
            batch_loss = example_losses.sum()
            loss = batch_loss

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            loss.backward()
            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in label)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += 1
            cum_examples += 1

            if step % 1000 == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, step,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.
            if step%1000==999:
                torch.save(model.state_dict(), args.save_path)
            

            # perform validation
        if epoch:
            print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, step,
                                                                                        cum_loss / cum_examples,
                                                                                        np.exp(cum_loss / cum_tgt_words),
                                                                                        cum_examples), file=sys.stderr)

            cum_loss = cum_examples = cum_tgt_words = 0.
            valid_num += 1

            print('begin validation ...', file=sys.stderr)

            # compute dev. ppl and bleu
            dev_ppl = evaluate_ppl(args, model, Subset(train_dataset,np.random.randint(0,len(train_dataset),1000)))   # dev batch size can be a bit larger
            valid_metric = -dev_ppl

            print('validation: iter %d, dev. ppl %f' % (epoch, dev_ppl), file=sys.stderr)

            is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
            hist_valid_scores.append(valid_metric)

            if is_better:
                patience = 0
                print('save currently the best model to [%s]' % args.save_path, file=sys.stderr)
                torch.save(model.state_dict(), args.save_path)
            elif patience < int(args.patience):
                patience += 1
                print('hit patience %d' % patience, file=sys.stderr)

                if patience == args.patience:
                    num_trial += 1
                    print('hit #%d trial' % num_trial, file=sys.stderr)
                    if num_trial == int(args.max_num_trial):
                        print('early stop!', file=sys.stderr)
                        exit(0)

                    # decay lr, and restore from previously best checkpoint
                    lr = optimizer.param_groups[0]['lr'] * float(args.lr_decay)
                    print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                    # load model
                    model.load_state_dict(torch.load(args.save_path))

                    # set new lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # reset patience
                    patience = 0

    print('reached maximum number of epochs!', file=sys.stderr)
    exit(0)

def beam_search(model, test_data_src, beam_size, max_decoding_time_step):
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)
    return hypotheses


def main(args):
    # seed the random number generators
    seed = args.seed
    torch.manual_seed(seed)
    if args.gpus[0] > -1:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    train(args)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in args.gpus])
    main(args)
