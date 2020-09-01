import torch
import json


class knowledgeLoader():
    def __init__(self):
        with open('/home/ailab/github/KnowledgeExtraction_ver2.0/dataloaders/knowledge.json', 'r') as f:
            self.knowledges = [json.loads(s) for s in f]
        self.knowledge_sentence = [item['text'] for item in self.knowledges]

    def get_knowledge_sentence(self):
        return self.knowledge_sentence

    def get_knowledge_items(self):
        return self.knowledges
