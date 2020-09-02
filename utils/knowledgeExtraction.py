from sentence_transformers import SentenceTransformer, util
import torch
import time
import json
from config import KNOWLEDGE_DIR
import os

class knowledgeExtraction():
    def __init__(self, file_name, keyword_list):
        with open(os.path.join(KNOWLEDGE_DIR,'{}.jsonl'.format(file_name)), 'r') as f:
            self.knowledges = [json.loads(s) for s in f]
        self.keyword_list = keyword_list

    def extract(self, sentence, knowledge_list):

        embedded_sentence = self.model.encode(sentence)
        embedded_knowledge_list = self.model.encode(knowledge_list)
        start = time.time()
        cos_sim = torch.topk(torch.t(embedded_sentence,embedded_knowledge_list))
        print('time : ',time.time-start)
        top_K = torch.topk(torch.t(cos_sim[0]),K)
        top_K_list = top_K[1].tolist()

        return top_K_list


    '''
    for i, concept in enumerate(concept_list):
    print(i)
    if concept['e1'] in hash_table.keys():
        new = hash_table[concept['e1']] + [{'e1':concept['e1'],'rel':concept['rel'],'e2':concept['e2'],'text':concept['text']}]
        hash_table[concept['e1']] = new
        continue
    hash_table[concept['e1']] = [{'e1':concept['e1'],'rel':concept['rel'],'e2':concept['e2'],'text':concept['text']}]
    '''