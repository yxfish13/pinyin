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
use_cuda = False
print(use_cuda)
MAX_LENGTH = 15

SOS_token = 0
EOS_token = 1
BLK_token = 2
hidden_size = 384
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


import numpy as np



def indexesFromSentence(lang, sentence):
    sentence = sentence.replace("\n","")
    res = [lang.word2index[word] for word in sentence.split(' ')]
    while len(res)<5:
        res.append(" ")
    if (len(res)>=15):
        res = res[0:14]
    return res

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

    return decoded_words#, decoder_attentions[:di + 1]

def translate(inputFile,encoder,decoder):
    ans = []
    with open(inputFile) as fin:
            while 1:
                line = fin.readline()
                if not line:break
                #print(line.replace("\n",""))
                mypys = evaluate(encoder,decoder,line.replace("\n",""))
                #print(mypys)
                ans.append("".join(mypys[:-1]))
    return ans

def outResult(inputFile,outputFile,encoder,decoder):
    result = translate(inputFile,encoder,decoder)
    #print(result)
    with open(outputFile,"w") as f:
        f.write("\n".join(result))


import pickle
input_lang  = pickle.load(open('input_lang.pkl', 'rb'))
output_lang = pickle.load(open('output_lang.pkl', 'rb'))


encoder = EncoderRNN(input_lang.n_words, hidden_size,n_layers=1)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words,n_layers=1
                               , dropout_p=0.1)
encoder.load_state_dict(torch.load('encoder1.pkl'))
attn_decoder.load_state_dict(torch.load('attn_decoder1.pkl'))

import sys
inputFile = sys.argv[1]
outputFile = sys.argv[2]


outResult(inputFile,outputFile,encoder,attn_decoder)