#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
import pickle
import time


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from model import sign_language_model
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
from apex import amp

parser = argparse.ArgumentParser(description='sl_model')
parser.add_argument(
    '--lr',
    type=float,
    default=3e-4,
    metavar='LR',
    help='learning rate (default: 0.0003)')
parser.add_argument(
    '--lr_decay',
    type=float,
    default=0.95,
    metavar='LR',
    help='learning rate (default: 0.95)')
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
    default=10086,
    metavar='S',
    help='random seed (default: 10086)')
parser.add_argument(
    '--train_src',
    type=str,
    default='./data/picture/',
    metavar='NS',
    help='the path of source picture (default:./data/picture/)')
parser.add_argument(
    '--vail_src',
    type=str,
    default='./data/picture/',
    metavar='NS',
    help='the path of source picture (default: ./data/picture/)')
parser.add_argument(
    '--max_epoch',
    type=int,
    default=20000,
    metavar='M',
    help='maximum length of an episode (default: 20000)')
parser.add_argument(
    '--vocab',
    default='./data/vocab/phoenix2014T.vocab.de',
    metavar='vo',
    help='The vocab path (default: ./data/vocab/phoenix2014T.vocab.de)')
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
    help='clip_grad (default: 5)')
parser.add_argument(
    '--save_path',
    type=str,
    default='./checkpoints/model_weather.pth',
    metavar='sp',
    help='the path to save the model (default:./checkpoints/model.pth)')
parser.add_argument(
    '--patience',
    type=int,
    default=5,
    metavar='pa',
    help='the patience to waiting the epoch alter')
parser.add_argument(
    '--max_num_trial',
    type=int,
    default=5,
    metavar='NT',
    help='(default: 5')
parser.add_argument(
    '--freeze',
    type=bool,
    default=False,
    metavar='FR',
    help='freeze the 3D conv(default: False')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    metavar='Bs',
    help='batch size(default: 1')
parser.add_argument(
    '--workers',
    type=int,
    default=10,
    metavar='W',
    help='how many training processes to use (default: 10)')
parser.add_argument(
    '--load', default=0,type=int, metavar='L', help='load a trained model')
parser.add_argument(
    '--save_step',
    default=10,
    metavar='SS',
    help='Save model num of epoch')
parser.add_argument(
    '--gpus',
    type=int,
    default=[0],
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')

parser.add_argument(
    '--bp_step',
    default=1,
    metavar='bps',
    help='the num of steps to accumulate the gard to back propagation (default: 8)')

parser.add_argument(
    '--loss_alpha',
    type=float,
    default=0.9,
    metavar='Al',
    help='The super parameter to balance between the lstm loss and ctc loss (default: 0.9)')

parser.add_argument(
    '--feature',
    type=str,
    default="R2Plus1D19",
    help='feature extractor for extracting image feature,P3D19P,R2Plus1D19,efficient')

parser.add_argument(
    '--fp16',
    type=bool,
    default=True,
    help='using fp16 to shrink the occupation of GPU (default True)'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
    print("model xavier initiation!")

def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    img_lens = []
    label_lens = []
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
            img_lens.append(len(batch[b][0]))
            label_lens.append(len(batch[b][1]))

            temp = torch.zeros_like(pad)
            temp[:len(batch[b][0]),:,:,:,:] = batch[b][0]
            images.append(temp)  # 已经对图片进行对齐操作
            labels.append(batch[b][1])

    images = torch.stack(images, 0)
    return images, labels, img_lens, label_lens


def evaluate_ppl(args, model, dev_dataset):
    """ Evaluate perplexity on dev sentences
    @param model (SL): sign language Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    vail_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,  
                            collate_fn=train_collate,
                            num_workers=args.workers)

    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for imgs, target, img_len, tar_len in tqdm(vail_loader):
            if args.gpus[0] > -1:
                imgs = imgs.cuda()

            example_losses, closs = model(imgs, target, img_len, tar_len)

            cum_loss += (0.1*example_losses.cpu().item() + 0.9*closs.cpu().item()) if not math.isinf(closs.item()) else example_losses.cpu().item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in target)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss)
        # ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()
    return ppl


# def compute_corpus_level_bleu_score(references, hypotheses):
#     """ Given decoding results and reference sentences, compute corpus-level BLEU score.
#     @param references (List[List[str]]): a list of gold-standard reference target sentences
#     @ param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
#     @returns bleu_score: corpus-level BLEU score
#     """
#     if references[0][0] == '<s>':
#         references = [ref[1:-1] for ref in references]
#     bleu_score = corpus_bleu([[ref] for ref in references],
#                              [hyp.value for hyp in hypotheses])
#     return bleu_score


def train(args):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """

    root_dir = "/data/shanyx/hrh/sign/weather/feature/"
    train_csv_file = "/data/shanyx/hrh/sign/weather/manual/PHOENIX.train.csv"
    vail_csv_file = "/data/shanyx/hrh/sign/weather/manual/PHOENIX.dev.csv"

    tf = transforms.Compose([
            transforms.Resize((260,210)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ])

    train_dataset = WeatherSignDataset(root_dir,train_csv_file,mode='train',num_frames_per_clip=8,transform=tf,skip_rate=4)
    # train_sample = random.sample(range(0,len(train_dataset)),2000)
    # train_dataset = Subset(train_dataset,train_sample)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,   #现在只能load一个batch，但是一个batch中有一组图片，对应一个翻译
                            shuffle=True,
                            collate_fn=train_collate,
                            num_workers=args.workers)
    
    vail_dataset = WeatherSignDataset(root_dir,vail_csv_file,mode='vail',num_frames_per_clip=8,transform=tf,skip_rate=4)

    clip_grad = args.clip_grad

    vocab = read_vocab_from_txt(args.vocab)


    model = sign_language_model(embed_size=args.embedding_size,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout,
                vocab=vocab,
                args=args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    #load model
    if args.load==1:
        print('load parameters of the model', file=sys.stderr)
        model.load_state_dict(torch.load(args.save_path))
    elif args.load==2:
        print('load p3d pretrain parameters of the model', file=sys.stderr)
        model.load_pretrain('./checkpoints/pretrain_encode.pth')
    else:
        weights_init(model)  
        
    if args.gpus[0] > -1:
        model = model.cuda()
        # torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”
    if len(args.gpus) > 1:
        model = nn.DataParallel(model)
        optimizer = nn.DataParallel(optimizer)
    num_trial = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    bp_loss_sum = torch.zeros((1,)).cuda()
    hist_valid_scores = []
    train_time = begin_time = time.time()
    log = Logger()
    log.open('./checkpoints/log_train_weather.txt', mode= 'a')
    print('begin Maximum Likelihood training')
    for epoch in range(args.max_epoch):
        step = 0
        for imgs, label, img_len, tar_len in tqdm(train_loader):
            # step: the index of data_loader
            # data: the return item of Class Dataset ,etc. (img_datas,img_label)
            # break
            step += 1
            if args.gpus[0] > -1:
                imgs = imgs.cuda().requires_grad_()

            optimizer.zero_grad()

            example_losses, closs = model(imgs, label, img_len, tar_len) # (batch_size) b-length value of the loss 
            # print(example_losses,closs)
            loss = 0.1*example_losses + 0.9*closs.sum() if not math.isinf(closs.item()) else example_losses
            batch_loss = loss
            # print(loss)
            # clip gradient
            if step%args.bp_step==0:
                bp_loss_sum+=loss
                if args.fp16:
                    with amp.scale_loss(bp_loss_sum, optimizer) as scaled_loss:
                        scaled_loss.backward()
                bp_loss_sum = torch.zeros((1,)).cuda()
            else:
                 bp_loss_sum += loss
            # loss.backward()

            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()
            
            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val
  
            tgt_words_num_to_predict = sum(len(s[1:]) for s in label)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += 1
            cum_examples += 1

            if step % 1000 == 0 :
                # print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                #       'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, step,
                #                                                                          report_loss / report_examples,
                #                                                                          math.exp(report_loss / report_tgt_words),
                #                                                                          cum_examples,
                #                                                                          report_tgt_words / (time.time() - train_time),
                #                                                                          time.time() - begin_time), file=sys.stderr)
                log.write('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec\n' % (epoch, step,
                                                                                         report_loss / report_examples,
                                                                                        #  math.exp(report_loss / report_tgt_words),
                                                                                         math.exp(report_loss/report_examples),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time)
                                                                                         )
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.
            if step%1000==999:
                torch.save(model.state_dict(), args.save_path)
            if step%500==499:
                torch.cuda.empty_cache()
            

        # perform validation
        if epoch:
            cum_loss = cum_examples = cum_tgt_words = 0.
            valid_num += 1

            print('begin validation ...', file=sys.stderr)

            # compute dev. ppl and bleu
            dev_ppl = evaluate_ppl(args, model, vail_dataset)   # dev batch size can be a bit larger
            valid_metric = -dev_ppl

            log.write('validation: iter %d, dev. ppl %f\n' % (epoch, dev_ppl))
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

# def beam_search(model, test_data_src, beam_size, max_decoding_time_step):
#     """ Run beam search to construct hypotheses for a list of src-language sentences.
#     @param model (sign): sign Model
#     @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
#     @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
#     @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
#     @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
#     """
#     model.eval()

#     hypotheses = []
#     with torch.no_grad():
#         for src_picts,__ in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
#             if model.args.gpus[0] > -1:
#                 src_picts = src_picts.cuda()
            
#             example_hyps = model.beam_search(src_picts, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
#             hypotheses.append(example_hyps)
#     return hypotheses


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
