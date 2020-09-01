from utils.topKknowledge import topKextraction
from dataloaders.knowledge_loader import knowledgeLoader

from sentence_transformers import SentenceTransformer, util


knowledge_loader = knowledgeLoader()
topKextraction = topKextraction(100)

sentence = "dog is cute."
knowledge_list = knowledge_loader.get_knowledge_sentence()

topKknowledge = topKextraction.topK(sentence, knowledge_list)
print(topKknowledge)

