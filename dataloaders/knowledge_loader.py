import torch
import json
from config import KNOWLEDGE_DIR
import os

class knowledgeLoader():
    def __init__(self, file_name):
        with open(os.path.join(KNOWLEDGE_DIR,'{}.jsonl'.format(file_name)), 'r') as f:
            self.knowledges = [json.loads(s) for s in f]

        self.knowledge_sentence = [item['text'] for item in self.knowledges]

    def get_knowledge_sentence(self):
        return self.knowledge_sentence

    def get_knowledge_items(self):
        return self.knowledges
