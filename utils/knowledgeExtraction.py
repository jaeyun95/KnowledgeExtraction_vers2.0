from sentence_transformers import SentenceTransformer, util
import torch
import time
import json
from config import KNOWLEDGE_DIR, EMBEDDING_DIR
import os
import random
import numpy as np
import h5py

class Hash():
    def __init__(self, file_name):
        self.hash_table = {}
        with open(os.path.join(KNOWLEDGE_DIR, '{}.json'.format(file_name)), 'r') as f:
            concept_list = [json.loads(s) for s in f]
        for i, concept in enumerate(concept_list):
            if concept['e1'] in self.hash_table.keys():
                new = self.hash_table[concept['e1']] + [
                    {'e1': concept['e1'], 'rel': concept['rel'], 'e2': concept['e2'], 'text': concept['text'], 'id':concept['id']}]
                self.hash_table[concept['e1']] = new
                continue
            self.hash_table[concept['e1']] = [
                {'e1': concept['e1'], 'rel': concept['rel'], 'e2': concept['e2'], 'text': concept['text'], 'id':concept['id']}]

    def get_hash_table(self):
        return self.hash_table

class KnowledgeExtraction():
    def __init__(self, limit, K, embedding_file=None):
        self.hash_table = Hash('knowledge').get_hash_table()
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self.limit = limit
        self.K = K
        self.embedding_file = embedding_file
        self.embedding_p5hy_path = os.path.join(EMBEDDING_DIR, f'{embedding_file}.h5')

    def _extract_hash(self, keywords):
        knowledges = []
        knowledges_text = []

        for i, keyword in enumerate(keywords):
            results = []
            if keyword in self.hash_table.keys():
                results = self.hash_table[keyword]
                random.shuffle(results)
                if self.limit < len(results):
                    results = results[:100]
            knowledges += results
        knowledges_text = [knowledge['text'] for knowledge in knowledges]
        return knowledges_text#knowledges, knowledges_text

    def _extract_topK(self, sentence, knowledges, index=None):
        if self.embedding_file == 'None':
            embedded_sentence = self.model.encode(sentence)
            embedded_knowledge_list = self.model.encode(knowledges)
        else:
            with h5py.File(self.embedding_p5hy_path, 'r') as h5:
                grp_items = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}
            embedded_sentence = grp_items['question']
            embedded_knowledge_list = grp_items['knowledges']
        cos_sim = util.pytorch_cos_sim(embedded_sentence, embedded_knowledge_list)[0]
        top_K = torch.topk(torch.t(cos_sim),self.K)
        top_K_list = top_K[1].tolist()
        return top_K_list

    def extract(self, keywords, sentence):
        extracted_knowledges, extracted_knowledges_text = self._extract_hash(keywords)
        extracted_topK = self._extract_topK(sentence, extracted_knowledges_text)
        return extracted_topK

