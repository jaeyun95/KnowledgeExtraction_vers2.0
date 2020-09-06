from dataloaders.vcr import VCR
from utils.keywordExtraction import KeywordExtraction
from utils.knowledgeExtraction import KnowledgeExtraction
from sentence_transformers import SentenceTransformer
import numpy as np
import h5py

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
keyword_extractor = KeywordExtraction()
knowledge_extractor = KnowledgeExtraction(50,50)

train_answer = VCR('train','answer')
train_rationale = VCR('train','rationale')
val_answer = VCR('val','answer')
val_rationale = VCR('val', 'rationale')

output_h5_answer = h5py.File(f'/media/ailab/songyoungtak/vcr1/val_answer.h5', 'w')
output_h5_rationale = h5py.File(f'/media/ailab/songyoungtak/vcr1/val_rationale.h5', 'w')

index = 0
for answer, rationale in zip(val_answer, val_rationale):
    print(index)
    output_h5_answer.create_group(f'{str(index)}')
    output_h5_rationale.create_group(f'{str(index)}')

    answer_question = answer['question_sentence']
    rationale_question = rationale['question_sentence']
    answer_keyword = [keyword_extractor.get_keyword(answer['answer_list'][i]+answer['objects']) for i in range(4)]
    rationale_keyword = [keyword_extractor.get_keyword(rationale['answer_list'][i]+answer['objects']) for i in range(4)]

    answer_knowledge = [knowledge_extractor._extract_hash(answer_keyword[i]) for i in range(4)]
    rationale_knowledge = [knowledge_extractor._extract_hash(rationale_keyword[i]) for i in range(4)]

    answer_question = model.encode(answer_question)
    rationale_question = model.encode(rationale_question)

    answer_knowledge = [model.encode(answer_knowledge[i]) for i in range(4)]
    rationale_knowledge = [model.encode(rationale_knowledge[i]) for i in range(4)]

    output_h5_answer[f'{str(index)}'].create_dataset(f'question', data=answer_question)
    output_h5_rationale[f'{str(index)}'].create_dataset(f'question', data=rationale_question)

    for i in range(4):
        output_h5_answer[f'{str(index)}'].create_dataset(f'knowledges'+str(i), data=np.array(answer_knowledge[i]))
        output_h5_rationale[f'{str(index)}'].create_dataset(f'knowledges'+str(i), data=np.array(rationale_knowledge[i]))
    index += 1
