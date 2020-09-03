from utils.knowledgeExtraction import KnowledgeExtraction
from utils.keywordExtraction import KeywordExtraction
from dataloaders.vcr import VCR
from sentence_transformers import SentenceTransformer, util



keyword_extractor = KeywordExtraction()
knowledge_extractor = KnowledgeExtraction(50,50)

keywords = keyword_extractor.get_keyword(["dog","is","cute","and","cat","is","cute","too."])
sentence = "dog is cute and cat is cute too."
knowledges = knowledge_extractor.extract(keywords, sentence)
'''
topKextraction = topKextraction(100)

sentence = "dog is cute."
knowledge_list = knowledge_loader.get_knowledge_sentence()

topKknowledge = topKextraction.topK(sentence, knowledge_list)
print(topKknowledge)




lists = []
for i,item in enumerate(knowledge):
    print(i)
    if item['text'] in lists:
        continue
    else:
        lists.append(item['text'])

print(len(lists))
'''