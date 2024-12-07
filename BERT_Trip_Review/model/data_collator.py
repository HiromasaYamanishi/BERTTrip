# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
import torch
import datetime
from torch.nn.utils.rnn import pad_sequence

from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from model.bert.bert_model import BertTripConfig

from util import dataset_metadata
import math
import pandas as pd
from sklearn.neighbors import BallTree

    
@dataclass
class DataCollatorForLanguageModeling:
    dataset: str
    poi_tokenizer: PreTrainedTokenizerBase
    review_tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    config: BertTripConfig = None
    stage: str = 'pretrain'
    mask: bool = True

    def __post_init__(self):
        if self.mlm and self.poi_tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        self.user_start_pos = 1
        self.user_end_pos = self.user_start_pos + self.config.num_user_token
        self.time_start_pos = self.user_end_pos
        self.time_end_pos = self.time_start_pos + self.config.num_time_token
        self.poi_type_id = 1
        self.review_type_id = 4
        self.user_type_id = 2
        self.time_type_id = 3
        self.ignore_mask_id = -100

    def __call__(
        self, 
        examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        poi_examples = []
        aug_poi_examples = []
        review_examples = []
        aug_review_examples = []

        for example in examples:
            #print('example', example)
            poi_examples.append({"input_ids": example['input_ids']})
            aug_poi_examples.append({"input_ids": example['aug_input_ids']})
            review_examples.append({"input_ids": example['review_ids']})
            aug_review_examples.append({"input_ids": example['aug_review_ids']})

        #print('before', review_examples)
        # パディング処理
        batch = self.poi_tokenizer.pad(poi_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        aug_batch = self.poi_tokenizer.pad(aug_poi_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        review_batch = self.review_tokenizer.pad(review_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        aug_review_batch = self.review_tokenizer.pad(aug_review_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        #print('after', review_examples)
        # バッチの構築
        batch['aug_input_ids'] = aug_batch['input_ids']
        batch['aug_attention_mask'] = aug_batch['attention_mask']
        batch['review_ids'] = review_batch['input_ids']
        batch['review_attention_mask'] = review_batch['attention_mask']
        batch['aug_review_ids'] = aug_review_batch['input_ids']
        batch['aug_review_attention_mask'] = aug_review_batch['attention_mask']

        if self.mask:
            # 共通のマスクパターンを生成
            special_tokens_mask = batch.pop("special_tokens_mask", None)
            masked_indices = self.generate_masked_indices(batch["input_ids"], special_tokens_mask)
            aug_masked_indices = self.generate_masked_indices(batch["aug_input_ids"], special_tokens_mask)

            # POIとreviewで同じマスクパターンを使用
            batch["input_ids"], batch["labels"], batch["aug_input_ids"], batch["aug_labels"] = self.mask_tokens(
                batch["input_ids"], batch['aug_input_ids'],
                tokenizer=self.poi_tokenizer,
                special_tokens_mask=special_tokens_mask,
                masked_indices=masked_indices,
                aug_masked_indices=aug_masked_indices,
                is_review=False
            )

            batch["review_ids"], batch["review_labels"], batch["aug_review_ids"], batch["aug_review_labels"] = self.mask_tokens(
                batch["review_ids"], batch['aug_review_ids'],
                tokenizer=self.review_tokenizer,
                special_tokens_mask=special_tokens_mask,
                masked_indices=masked_indices,  # 同じマスクパターン
                aug_masked_indices=aug_masked_indices,  # 同じマスクパターン
                is_review=True
            )
        else:
            batch["labels"] = batch["input_ids"].clone()
            batch["aug_labels"] = batch["aug_input_ids"].clone() if self.config.use_data_agumentation else None
            batch["review_labels"] = batch["review_ids"].clone()
            batch["aug_review_labels"] = batch["aug_review_ids"].clone() if self.config.use_data_agumentation else None

        if not self.config.use_data_agumentation:
            keys_to_remove = ['aug_input_ids', 'aug_labels', 'aug_review_ids', 'aug_review_labels']
            for key in keys_to_remove:
                if key in batch:
                    del batch[key]

        batch['stage'] = self.stage
        if self.stage == 'pretrain':
            batch['input_ids'] = torch.cat([batch['input_ids'], torch.ones((batch['input_ids'].size(0), 1))*10000], axis=1)
        else:
            batch['input_ids'] = torch.cat([batch['input_ids'], torch.ones((batch['input_ids'].size(0), 1))*99999], axis=1)

        return batch

    def generate_masked_indices(self, inputs, special_tokens_mask):
        """共通のマスクパターンを生成"""
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.poi_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in inputs.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        if self.config.use_data_agumentation:
            return self.data_agumentation(inputs, special_tokens_mask, -1)
        else:
            return self.data_agumentation(inputs, special_tokens_mask, 0.15)

    def data_agumentation(self, labels, special_tokens_mask, mln_probability=0.15):
        if mln_probability == -1:
            mln_probability = torch.rand(1).uniform_(0.15, 0.5)[0]
        probability_matrix = torch.full(labels.shape, mln_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        return masked_indices

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        aug_inputs: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        special_tokens_mask: Optional[torch.Tensor] = None,
        masked_indices: Optional[torch.Tensor] = None,
        aug_masked_indices: Optional[torch.Tensor] = None,
        is_review: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        USER_NUM = dataset_metadata[self.dataset]['USER_NUM']
        TIME_NUM = dataset_metadata[self.dataset]['TIME_NUM']
        TIME_START_INDEX = len(tokenizer) - USER_NUM - TIME_NUM
        USER_START_INDEX = TIME_START_INDEX + TIME_NUM

        labels = inputs.clone()
        input_types = torch.full(labels.shape, self.review_type_id if is_review else self.poi_type_id)
        if config.add_user_token:
            input_types[:, self.user_start_pos:self.user_end_pos] = self.user_type_id
        if config.add_time_token:
            input_types[:, self.time_start_pos:self.time_end_pos] = self.time_type_id

        labels[~masked_indices] = self.ignore_mask_id

        # POI/Reviewのマスク処理
        is_target = (input_types == (self.review_type_id if is_review else self.poi_type_id))
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices & is_target
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & is_target & ~indices_replaced
        inputs_random_words = torch.randint(TIME_START_INDEX, labels.shape, dtype=torch.long)
        inputs[indices_random] = inputs_random_words[indices_random]

        # User/Timeのマスク処理
        if config.add_user_token:
            is_user = (input_types == self.user_type_id)
            labels[is_user] = self.ignore_mask_id
            user_indices_random = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices & is_user
            inputs_random_words = torch.randint(USER_START_INDEX, USER_START_INDEX + USER_NUM, labels.shape, dtype=torch.long)
            inputs[user_indices_random] = inputs_random_words[user_indices_random]

        if config.add_time_token:
            is_time = (input_types == self.time_type_id)
            labels[is_time] = self.ignore_mask_id
            time_indices_random = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices & is_time
            inputs_random_words = torch.randint(TIME_START_INDEX, TIME_START_INDEX + TIME_NUM, labels.shape, dtype=torch.long)
            inputs[time_indices_random] = inputs_random_words[time_indices_random]

        # Augmentationの処理
        if config.use_data_agumentation:
            aug_inputs, aug_labels = self._process_augmentation(
                tokenizer,
                aug_inputs.clone(),
                input_types,
                aug_masked_indices,
                TIME_START_INDEX,
                USER_START_INDEX,
                USER_NUM,
                TIME_NUM,
                is_review
            )
        else:
            aug_inputs, aug_labels = None, None

        return inputs, labels, aug_inputs, aug_labels

    def _process_augmentation(
        self,
        tokenizer,
        aug_inputs,
        input_types,
        masked_indices,
        TIME_START_INDEX,
        USER_START_INDEX,
        USER_NUM,
        TIME_NUM,
        is_review
    ):
        aug_labels = aug_inputs.clone()
        aug_labels[~masked_indices] = self.ignore_mask_id

        is_target = (input_types == (self.review_type_id if is_review else self.poi_type_id))
        indices_replaced = torch.bernoulli(torch.full(aug_labels.shape, 0.8)).bool() & masked_indices & is_target
        aug_inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(aug_labels.shape, 0.5)).bool() & masked_indices & is_target & ~indices_replaced
        aug_inputs_random_words = torch.randint(TIME_START_INDEX, aug_labels.shape, dtype=torch.long)
        aug_inputs[indices_random] = aug_inputs_random_words[indices_random]

        if self.config.add_user_token:
            is_user = (input_types == self.user_type_id)
            aug_labels[is_user] = self.ignore_mask_id
            user_indices_random = torch.bernoulli(torch.full(aug_labels.shape, 1.0)).bool() & masked_indices & is_user
            aug_inputs_random_words = torch.randint(USER_START_INDEX, USER_START_INDEX + USER_NUM, aug_labels.shape, dtype=torch.long)
            aug_inputs[user_indices_random] = aug_inputs_random_words[user_indices_random]

        if self.config.add_time_token:
            is_time = (input_types == self.time_type_id)
            aug_labels[is_time] = self.ignore_mask_id
            time_indices_random = torch.bernoulli(torch.full(aug_labels.shape, 1.0)).bool() & masked_indices & is_time
            aug_inputs_random_words = torch.randint(TIME_START_INDEX, TIME_START_INDEX + TIME_NUM, aug_labels.shape, dtype=torch.long)
            aug_inputs[time_indices_random] = aug_inputs_random_words[time_indices_random]

        return aug_inputs, aug_labels
    
@dataclass
class DataCollatorForLanguageModeling_:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_specfial_tokens_mask=True`.
    """
    dataset: str
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    config: BertTripConfig = None
    stage: str = 'pretrain'
    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        self.user_start_pos = 1
        self.user_end_pos = self.user_start_pos + self.config.num_user_token
        self.time_start_pos = self.user_end_pos
        self.time_end_pos = self.time_start_pos + self.config.num_time_token
        self.poi_type_id = 1
        self.user_type_id = 2
        self.time_type_id = 3
        self.ignore_mask_id = -100
        
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        poi_examples = []
        aug_poi_examples = []

        for example in examples:
            poi_examples.append({"input_ids": example['input_ids']})
            aug_poi_examples.append({"input_ids": example['aug_input_ids']})
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(poi_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch['aug_input_ids'] = self.tokenizer.pad(aug_poi_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch['aug_attention_mask'] = batch['aug_input_ids']["attention_mask"]
        batch['aug_input_ids'] = batch['aug_input_ids']["input_ids"]

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"], batch["aug_input_ids"], batch["aug_labels"]= self.mask_tokens(
            batch["input_ids"], batch['aug_input_ids'], special_tokens_mask=special_tokens_mask
        )

        if not self.config.use_data_agumentation:
            if 'aug_input_ids' in batch:
                del batch['aug_input_ids']
            if 'aug_labels' in batch:
                del batch['aug_labels']

        batch['stage'] = self.stage #[self.stage for _ in range(len(poi_examples))]
        batch['input_ids'] = batch['input_ids']
        if self.stage == 'pretrain':
            batch['input_ids'] = torch.cat([batch['input_ids'], torch.ones((batch['input_ids'].size(0), 1))*10000], axis=1)
        else:
            batch['input_ids'] = torch.cat([batch['input_ids'], torch.ones((batch['input_ids'].size(0), 1))*99999], axis=1)
        # print('batch stage', batch['stage'], batch)
        return batch

    def data_agumentation(self, labels, special_tokens_mask, mln_probability = 0.15):
         if mln_probability == -1:
             mln_probability = torch.rand(1).uniform_(0.15, 0.5)[0]
         probability_matrix = torch.full(labels.shape, mln_probability)
         probability_matrix.masked_fill_(special_tokens_mask, value = 0.0)
         masked_indices = torch.bernoulli(probability_matrix).bool()
         return masked_indices

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        aug_inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        config = self.config
        USER_NUM = dataset_metadata[self.dataset]['USER_NUM']
        TIME_NUM = dataset_metadata[self.dataset]['TIME_NUM']
        # print('tokenizer', self.tokenizer)
        # print('len tokenizer', len(self.tokenizer))
        TIME_START_INDEX = len(self.tokenizer) - USER_NUM - TIME_NUM
        USER_START_INDEX = TIME_START_INDEX + TIME_NUM

        labels = inputs.clone()
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        input_types = torch.full(labels.shape, self.poi_type_id)
        if config.add_user_token:
            input_types[:, self.user_start_pos:self.user_end_pos] = self.user_type_id
        if config.add_time_token:
            input_types[:, self.time_start_pos:self.time_end_pos] = self.time_type_id

        if config.use_data_agumentation:
            masked_indices = self.data_agumentation(labels, special_tokens_mask, -1)
        else:
            masked_indices = self.data_agumentation(labels, special_tokens_mask, 0.15)
        labels[~masked_indices] = self.ignore_mask_id
        # print('input', input_types)
        # print('inputs', labels)
        #input
        is_poi = (input_types == self.poi_type_id)
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices & is_poi
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & is_poi & ~indices_replaced
        #print('random word', TIME_START_INDEX, labels.shape)
        inputs_random_words = torch.randint(TIME_START_INDEX, labels.shape, dtype=torch.long)
        inputs[indices_random] = inputs_random_words[indices_random]

        # user
        if config.add_user_token:
            is_user = (input_types == self.user_type_id)
            labels[is_user] = self.ignore_mask_id
            user_indices_random = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices & is_user
            inputs_random_words = torch.randint(USER_START_INDEX, USER_START_INDEX + USER_NUM, labels.shape, dtype=torch.long)
            inputs[user_indices_random] = inputs_random_words[user_indices_random]

        # time
        if config.add_time_token:
            is_time = (input_types == self.time_type_id)
            labels[is_time] = self.ignore_mask_id
            indices_random = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices & is_time
            inputs_random_words = torch.randint(TIME_START_INDEX, TIME_START_INDEX + TIME_NUM, labels.shape, dtype=torch.long)
            inputs[indices_random] = inputs_random_words[indices_random]


        if config.use_data_agumentation:
            aug_inputs = aug_inputs.clone()
            aug_labels = aug_inputs.clone()
            aug_special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in aug_labels.tolist()]
            aug_special_tokens_mask = torch.tensor(aug_special_tokens_mask, dtype=torch.bool)
            aug_input_types = torch.full(aug_labels.shape, self.poi_type_id)
            if config.add_user_token:
                aug_input_types[:, self.user_start_pos:self.user_end_pos] = self.user_type_id
            if config.add_time_token:
                aug_input_types[:, self.time_start_pos:self.time_end_pos] = self.time_type_id

            aug_masked_indices = self.data_agumentation(aug_labels, aug_special_tokens_mask, -1)
            aug_labels[~aug_masked_indices] = self.ignore_mask_id

            #input
            aug_is_poi = (aug_input_types == self.poi_type_id)
            aug_indices_replaced = torch.bernoulli(torch.full(aug_labels.shape, 0.8)).bool() & aug_masked_indices & aug_is_poi
            aug_inputs[aug_indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            aug_indices_random = torch.bernoulli(torch.full(aug_labels.shape, 0.5)).bool() & aug_masked_indices & aug_is_poi & ~aug_indices_replaced
            aug_inputs_random_words = torch.randint(TIME_START_INDEX, aug_labels.shape, dtype=torch.long)
            aug_inputs[aug_indices_random] = aug_inputs_random_words[aug_indices_random]

            #user
            if config.add_user_token:
                aug_is_user = (aug_input_types == self.user_type_id)
                aug_labels[aug_is_user] = self.ignore_mask_id
                aug_indices_random = torch.bernoulli(torch.full(aug_labels.shape, 1.0)).bool() & aug_masked_indices & aug_is_user
                aug_inputs_random_words = torch.randint(USER_START_INDEX, USER_START_INDEX + USER_NUM, aug_labels.shape, dtype=torch.long)
                aug_inputs[aug_indices_random] = aug_inputs_random_words[aug_indices_random]

            #time
            if config.add_time_token:
                aug_is_time = (aug_input_types == self.time_type_id)
                aug_labels[aug_is_time] = self.ignore_mask_id
                aug_indices_random = torch.bernoulli(torch.full(aug_labels.shape, 1.0)).bool() & aug_masked_indices & aug_is_time
                aug_inputs_random_words = torch.randint(TIME_START_INDEX, TIME_START_INDEX + TIME_NUM, aug_labels.shape, dtype=torch.long)
                aug_inputs[aug_indices_random] = aug_inputs_random_words[aug_indices_random]
        else:
            aug_inputs = None
            aug_labels = None
        return inputs,  labels, aug_inputs, aug_labels
