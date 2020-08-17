import spacy
nlp = spacy.load('en_core_web_sm')

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab
    
def emmbeding_data(vocab):
    dic={}
    lis=list(vocab)
    for x in lis:
        doc=nlp(x)
        dic[x]=doc.vector
    return dic        

import torch
def convert_to_tensor_representation(embedding_dic,data):
    return_list=[]
    for x in data:
        x_list=[]
        x_word=x[0]
        for y in x_word:
            vector=embedding_dic[y]
            x_list.append(vector)
        tensor=torch.FloatTensor(x_list)
        list_size=tensor.size()
        rechange_tensor=tensor.view(list_size[0],1,list_size[1])
        return_list.append(rechange_tensor)
    return return_list
