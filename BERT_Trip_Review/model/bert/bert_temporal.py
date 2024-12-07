# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from model.bert.bert_model import BertForMaskedLM, BertModel, BertOnlyMLMHead, MaskedLMOutput

def extract_poi_ids_and_index(file_path: str) -> tuple[list, int, int]:
   """
   Extract POI IDs from vocabulary file and get start/end indices
   
   Args:
       file_path (str): Path to vocabulary file
       
   Returns:
       tuple[list, int, int]: 
           - List of POI IDs as integers
           - Start index (index after [UNK])
           - End index (index before first time token)
   """
   poi_ids = []
   start_idx = -1
   end_idx = -1
   
   with open(file_path, 'r') as f:
       lines = f.readlines()
       
       # Find [UNK] and time- positions
       for i, line in enumerate(lines):
           line = line.strip()
           if line == '[UNK]':
               start_idx = i + 1
           elif line.startswith('time-'):
               end_idx = i
               break
       
       # Extract POI IDs
       if start_idx != -1 and end_idx != -1:
           for line in lines[start_idx:end_idx]:
               if line.strip().isdigit():
                   poi_ids.append(int(line.strip()))
                   
   return poi_ids, start_idx, end_idx

class BertEmbeddings_(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + config.num_extra_tokens, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "relative_key_query")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings + config.num_extra_tokens).expand((1, -1)))

    def forward(
        self,
        input_ids = None,
        token_type_ids = None,
        time_ids=None,
        aug_time_ids=None,
        position_ids = None,
        past_key_values_length = 0,
        inputs_embeds = None
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        inputs_embeds = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(embeddings)

        return embeddings
    
class BertEmbeddings(nn.Module):
    def __init__(self, config, mode, load_pretrain):
        super().__init__()
        if mode=='poi':
            self.word_embeddings = nn.Embedding(config.poi_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        elif mode=='review':
            self.word_embeddings = nn.Embedding(config.review_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + config.num_extra_tokens, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "relative_key_query")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings + config.num_extra_tokens).expand((1, -1)))
        if load_pretrain:
            if mode == 'poi':
                pretrain_embeddings = torch.load(config.poi_review_emb_path)
                poi_ids, start_idx, end_idx = extract_poi_ids_and_index(config.poi_vocab_file)
                #print('poi', poi_ids, start_idx, end_idx, torch.stack([torch.mean(pretrain_embeddings[p], dim=0) for p in poi_ids]).shape, self.word_embeddings.weight.shape)
                pretrain_embeddings_ordered = torch.stack([torch.mean(pretrain_embeddings[p], dim=0) for p in poi_ids])
            elif mode == 'review':
                pretrain_embeddings = torch.load(config.review_emb_path)
                review_ids, start_idx, end_idx = extract_poi_ids_and_index(config.review_vocab_file)
                #print('review', review_ids, start_idx, end_idx, torch.stack([pretrain_embeddings[p] for p in review_ids]).shape, self.word_embeddings.weight.shape)
                pretrain_embeddings_ordered = torch.stack([pretrain_embeddings[p] for p in review_ids])

                # 事前学習済み埋め込みを代入
            with torch.no_grad():
                self.word_embeddings.weight[start_idx:end_idx] = pretrain_embeddings_ordered

            # 指定範囲の重みを学習不可に設定
            self.word_embeddings.weight.register_hook(lambda grad: self._zero_grad_hook(grad, start_idx, end_idx))

    def _zero_grad_hook(self, grad, start_idx, end_idx):
        """指定範囲の勾配をゼロにするフック"""
        grad[start_idx:end_idx] = 0
        return grad

    def forward(
        self,
        input_ids = None,
        token_type_ids = None,
        time_ids=None,
        aug_time_ids=None,
        position_ids = None,
        past_key_values_length = 0,
        inputs_embeds = None
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        inputs_embeds = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertTemporalModel(BertModel):
    def __init__(self, config, add_pooling_layer=True, mode='poi', load_pretrain=False):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config, mode, load_pretrain)
        self.init_weights()

class TemporalBERT(BertForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.config = config
        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.bert = BertTemporalModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids = None,
        labels=None,
        attention_mask=None,
        aug_input_ids = None,
        aug_labels= None,
        aug_attention_mask=None,
        token_type_ids=None,
        time_ids=None,
        aug_time_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        ghash_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target_index = None,
        traj_ids = None,
        timestamps=None,
        aug_timestamps=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        z1_representation = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            time_ids=time_ids,
            aug_time_ids=aug_time_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        head1 = self.cls(z1_representation)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(head1.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            loss = None

        return MaskedLMOutput(
            loss=loss,
            logits=head1,
        )
