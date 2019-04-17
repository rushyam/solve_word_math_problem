#!/usr/bin/env python
# coding: utf-8

# In[49]:


import torch
import io
import numpy as np
import spacy
import math
import json
import random
from spacy.lemmatizer import Lemmatizer
from gensim.models.fasttext import FastText
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[50]:


word_vec_dim = 100
seq_len = 25


# In[51]:


def random_data(data):
    count = 0
    while(count < 500):
        x = random.randint(0,len(data)-1)
        y = random.randint(0,len(data)-1)
        temp = data[x]
        data[x] = data[y]
        data[y] = temp
        count += 1
    return data


# In[52]:


def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)


# In[53]:


def find_operation(eq):
    for o in eq:
        if(o == '+' or o == '-' or o == '*' or o == '/'):
            return o


# In[54]:


def chunck(question):
    question = question.lower()
    question = question.replace("’s", "")
    question = question.replace("'s", "")
    question = lemmatizer(question)
    question_word = []
    numerical_quantity = []
    doc = nlp(question)
    for token in doc:
        if token.pos_ != 'NUM' and token.pos_ != 'NOUN' and token.pos_ != 'DET' and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and token.text != '-PRON-':
            question_word.append(token.text)
    return question_word


# In[55]:


def load_vectors(fname):
    data = {}
    file = open(fname, 'r')
    for line in file:
        tokens = line.split(' ')
        X = tokens[1:]
        X = np.array(X)
        X[X == ''] = 0.0
        X = X.astype(np.float)
        data[tokens[0]] = X 
    return data


# In[56]:


def similarity(w1, w2):
    x = word2vec[w1]
    y = word2vec[w2]
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))


# In[135]:


def read(batch):
    X = np.zeros((seq_len, 10, word_vec_dim))
    Y = np.zeros((10, 4))
    one_hot = {
        '+' : 0,
        '-' : 1, 
        '*' : 2,
        '/' : 3
    }
    ques = []
    for q in range(10):
        ques.append(chunck(data[batch*10+q]['sQuestion']))
        ans = find_operation(data[batch*10+q]['lEquations'][0])
        Y[q][one_hot[ans]] = 1.0

    for w in range(seq_len):
        for q in range(10):
            if len(ques[q]) > w and ques[q][w] in word2vec:
                v = word2vec.get(ques[q][w])
                for i in range(word_vec_dim):
                    X[w][q][i] = v[i]
    
    y = torch.from_numpy(Y).long()
    x = torch.from_numpy(X).float()
    return x, y


# In[160]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size = word_vec_dim, hidden_size = 100, num_layers = 2)
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        
    def forward(self, x):
        o, z = self.lstm(x)
        x = self.fc1(z[-1][1])
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x


# In[139]:


with open('./DATA/singleop.json') as file:
    data = json.load(file)
data = random_data(data)
    
nlp = spacy.load("en")

word2vec = load_vectors('./glove.6B/glove.6B.100d copy.txt')


# In[161]:


model = Model()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# In[162]:


for epoch in range(5):
    for batch in range(50):
        inputs, labels = read(batch)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(batch, epoch)

print('done')