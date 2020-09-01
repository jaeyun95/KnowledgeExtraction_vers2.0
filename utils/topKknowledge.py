from sentence_transformers import SentenceTransformer, util
import torch


class topKextraction():
    def __init__(self, K):
        self.K = K
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    def topK(self, sentence, knowledge_list):

        embedded_sentence = self.model.encode(sentence)
        embedded_knowledge_list = self.model.encode(knowledge_list)

        cos_sim = torch.topk(torch.t(embedded_sentence,embedded_knowledge_list))
        top_K = torch.topk(torch.t(cos_sim[0]),5)
        top_K_list = top_K[1].tolist()

        return top_K_list