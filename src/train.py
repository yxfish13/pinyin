from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 15

SOS_token = 0
EOS_token = 1
BLK_token = 2
BatchSize = 64

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS",2: " "}
        self.n_words = 3  # Count SOS and EOS and BLK_token

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #embedded = self.embedding(input).view(1, BatchSize, -1)
        #print(input)
        embedded = self.embedding(input)
        #print(embedded.size())
        output = embedded
        for i in range(self.n_layers):
        #    print(output.size(),hidden.size())
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, BatchSize, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.2, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #embedded = self.embedding(input).view(1, BatchSize, -1)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), 1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0).permute(1,0,2),
                                 encoder_outputs).permute(1,0,2)
        #print("bmm result",attn_applied)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
teacher_forcing_ratio = 0.5

pinyin =[]
hanzi =[]
import codecs
with codecs.open("py_hanzi.txt","rb",'utf-8') as fin:
    inputdata = fin.readlines()
    pinyin = [line.split('\t')[0] for line in  inputdata]
    hanzi  = [line.split('\t')[1] for line in  inputdata]


def prepareData(lang1, lang2, reverse=False):
    input_lang=Lang("pinyin")
    output_lang=Lang("hanzi")
    pairs=[]
    for idx in range(len(pinyin[0:])):
        if (len(lang1[idx].split(" ")) >= 5 and len(lang1[idx].split(" "))<MAX_LENGTH):
            s1  = lang1[idx]#.split(" ")
            s2 = lang2[idx].replace("\n","")#.split(" ")
            pairs.append((s1,s2))
            input_lang.addSentence(s1)
            output_lang.addSentence(s2)
            
    return input_lang, output_lang, pairs
import pickle



input_lang,output_lang,pairs = prepareData(pinyin,hanzi)

pickle.dump(input_lang,open('input_lang.pkl', 'wb'))
pickle.dump(output_lang,open('output_lang.pkl', 'wb'))
print(len(pairs))

import numpy as np



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def aligneindex(indexes,MAX_LENGTH):
    return indexes+[BLK_token]*(MAX_LENGTH-len(indexes))
def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    indexes = aligneindex(indexes,MAX_LENGTH)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(input_lang,output_lang,pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #print("---------train---------")
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(BatchSize,max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        #print(encoder_output.size())
        encoder_outputs[:,ei] = encoder_output[0]
    #print(encoder_outputs)
    #return 
    decoder_input = Variable(torch.LongTensor([[SOS_token]*BatchSize]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #use_teacher_forcing = False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #print("---------decoder---------")
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #print(decoder_output)
            #print(target_variable[di])
            loss += criterion(decoder_output, target_variable[di].view(-1))
            decoder_input = target_variable[di]  # Teacher forcing
            #print(decoder_input)
            #return 
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            #print(topi)
            decoder_input = Variable(topi.view(1,-1))
            #print(decoder_input)
            #print(topi.cpu().max())
            #return 
            
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, target_variable[di].view(-1))
            if topi.cpu().max()<3:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=6e-3):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    #training_pairs = [variablesFromPair(input_lang,output_lang,random.choice(pairs))
    #                  for i in range(n_iters*BatchSize)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        #print("in loop")
        input_variable = []
        target_variable = []
        
        training_pairBatch = [variablesFromPair(input_lang,output_lang,pairs[random.randint(0,1000000)])#len(pairs)-1)])
                      for i in range(BatchSize)]
        for training_pair in training_pairBatch:
            input_variable.append(training_pair[0].data.numpy())
            target_variable.append(training_pair[1].data.numpy())
        #print(input_variable)
        input_variable = torch.LongTensor(np.asarray(input_variable)).permute(1,2,0)
        target_variable = torch.LongTensor(np.asarray(target_variable)).permute(1,2,0)
        input_variable = Variable(input_variable.cuda())
        target_variable =Variable(target_variable.cuda())
        #print(input_variable.size())
        #print(target_variable.size())
        #return 
        #print(input_variable.size())
        #return
        #print(input_variable.size(),target_variable.size())
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        #print("trainover")
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % 2000 ==0 :
            torch.save(encoder1.state_dict(), 'encoder.pkl')
            torch.save(attn_decoder1.state_dict(), 'attn_decoder.pkl')
            evaluateRandomly(encoder1, attn_decoder1, n=10)
        
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    #print("init",encoder.initHidden())
    encoder_hidden = encoder.initHidden()[:,0,:].unsqueeze(0)

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
    for ei in range(input_length):
        #print("test",encoder_hidden.size())
        encoder_output, encoder_hidden = encoder(input_variable[ei].view(1,1),
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs.unsqueeze(0))
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        #print(topi)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = pairs[random.randint(0,1000000)]
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
hidden_size = 384
BatchSize = 4096

print("begin trainning")
encoder1 = EncoderRNN(input_lang.n_words, hidden_size,n_layers=1)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,n_layers=1
                               , dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, 20000, print_every=30)
