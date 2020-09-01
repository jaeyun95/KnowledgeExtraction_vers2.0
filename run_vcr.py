from dataloaders.vcr import VCR
import torch

train_answer = VCR('train','answer')
train_rationale = VCR('train','rationale')
val_answer = VCR('val','answer')
val_rationale = VCR('val', 'rationale')
#test_answer = VCR('test','answer')
#test_rationale = VCR('test', 'rationale')


#for i,question in enumerate(train_answer):
