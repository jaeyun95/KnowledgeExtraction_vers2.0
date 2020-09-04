#from utils.knowledgeExtraction import KnowledgeExtraction
#from utils.keywordExtraction import KeywordExtraction
#from dataloaders.vcr import VCR
#from sentence_transformers import SentenceTransformer, util

import h5py
import numpy as np
'''
keyword_extractor = KeywordExtraction()
knowledge_extractor = KnowledgeExtraction(50,50)

keywords = keyword_extractor.get_keyword(["dog","is","cute","and","cat","is","cute","too."])
sentence = "dog is cute and cat is cute too."
knowledges = knowledge_extractor.extract(keywords, sentence)


//
topKextraction = topKextraction(100)

sentence = "dog is cute."
knowledge_list = knowledge_loader.get_knowledge_sentence()

topKknowledge = topKextraction.topK(sentence, knowledge_list)
print(topKknowledge)
///



lists = []
for i,item in enumerate(knowledge):
    print(i)
    if item['text'] in lists:
        continue
    else:
        lists.append(item['text'])

print(len(lists))
'''

with h5py.File('/media/ailab/jaeyun/VCR_knowledges/knowledges_embedding.h5', 'r') as h5:
    grp_items = {k: np.array(v, dtype=np.float16) for k, v in h5['knowledge'].items()}
print(grp_items)