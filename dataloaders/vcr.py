"""
Dataloaders for VCR
"""
import json
import os
import numpy as np
import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
import h5py
from copy import deepcopy
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

def _fix_tokenization(tokenized_sent, old_det_to_new_ind, obj_to_type):
    new_tokenization_with_tags = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = obj_to_type[int_name]
                new_ind = old_det_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("Oh no, the new index is negative! that means it's invalid. {} {}".format(
                        tokenized_sent, old_det_to_new_ind
                    ))
                text_to_use = GENDER_NEUTRAL_NAMES[
                    new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                new_tokenization_with_tags.append(text_to_use)
        else:
            new_tokenization_with_tags.append(tok)
    return new_tokenization_with_tags

class VCR(Dataset):
    def __init__(self, split, mode, only_use_relevant_dets=True, add_image_as_a_box=True):
        self.only_use_relevant_dets = only_use_relevant_dets
        self.mode = mode
        self.split = split
        self.add_image_as_a_box = add_image_as_a_box
        with open(os.path.join(VCR_ANNOTS_DIR,'{}.jsonl'.format(split)), 'r') as f:
            self.items = [json.loads(s) for s in f]

        self.token_indexers = {'elmo': ELMoTokenCharactersIndexer()}
        self.vocab = Vocabulary()

        with open(os.path.join(os.path.dirname(VCR_ANNOTS_DIR), 'cocoontology.json'), 'r') as f:
            coco = json.load(f)

        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}

    def __len__(self):
        return len(self.items)

    def _get_dets_to_use(self, item, check, sentence):
        if check == 'question': sentence = sentence
        else: sentence = sentence
        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item['objects']), dtype=bool)
            people = np.array([x == 'person' for x in item['objects']], dtype=bool)
            for sent in sentence:
                if isinstance(sent, list):
                    for tag in sent:
                        if tag >= 0 and tag < len(item['objects']):  # sanity check
                            dets2use[tag] = True
                elif sent.lower() in ('everyone', 'everyones'):
                    dets2use |= people
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(item['objects']), dtype=bool)
        dets2use = np.where(dets2use)[0]
        old_det_to_new_ind = np.zeros(len(item['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind

    def __getitem__(self, index):
        item = deepcopy(self.items[index])
        instance = {}

        if self.mode == 'rationale' and self.split != 'test':
            conditioned_label = item['answer_label']
            item['question'] += item['answer_choices'][conditioned_label]

        # question convert
        q_dets2use, q_old_det_to_new_ind = self._get_dets_to_use(item,'question',item['question'])

        # question
        instance['question'] = _fix_tokenization(item['question'],q_old_det_to_new_ind,item['objects'])

        # answer convert
        if self.mode == 'rationale': answer_list = item['rationale_choices']
        else: answer_list = item['answer_choices']
        a_dets2use, a_old_det_to_new_ind = [], []
        for k in range(4):
            pre_a_dets2use, pre_a_old_det_to_new_ind = self._get_dets_to_use(item,'answer',answer_list[k])
            a_dets2use.append(pre_a_dets2use)
            a_old_det_to_new_ind.append(pre_a_old_det_to_new_ind)
            answer_list[k] = _fix_tokenization(answer_list[k],a_old_det_to_new_ind[k],item['objects'])

        # answer
        instance['answer_list'] = answer_list

        #objects
        instance['objects'] = item['objects']

        return instance
