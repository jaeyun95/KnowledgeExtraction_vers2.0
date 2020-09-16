from utils.knowledgeExtraction import KnowledgeExtraction
from utils.keywordExtraction import KeywordExtraction
from dataloaders.vcr import VCR
from sentence_transformers import SentenceTransformer, util

keyword_extractor = KeywordExtraction()
knowledge_extractor = KnowledgeExtraction(50,50)

## set your sentence
sentence = "doc is cute and cat is cute too."

## extract keyword from your sentence(tokenized)
tokenized_sentence = sentence.split(' ')

## extract keyword
keywords = keyword_extractor.get_keyword(tokenized_sentence)

## extract knowledges
knowledges = knowledge_extractor.extract(keywords, sentence)

## extract topK knowledges
topKknowledge = knowledge_extractor._extract_topK(sentence, knowledges)




