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
from model.bert.bert_model import BertForMaskedLM, BertModel, BertOnlyMLMHead, MaskedLMOutput, BertOnlyCombineMLMHead
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.l2 = nn.Linear(hidden_dim, in_dim)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class ReviewTripBERT(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # ベースとなる3つのBERTモデル
        self.bert = BertModel(config, add_pooling_layer=True)  # 行動特徴量用
        self.review_bert = BertModel(config, add_pooling_layer=True)  # レビュー列特徴量用
        self.target_bert = BertModel(config, add_pooling_layer=True)  # ターゲットレビュー用
        
        # Attention層の定義
        self.review_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        # 投影層
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # self.context_projector = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.LayerNorm(config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.hidden_size)
        # )

        # self.review_projector = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.LayerNorm(config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.hidden_size)
        # )

        # self.target_projector = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.LayerNorm(config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.hidden_size)
        # )
        self.projector = projection_MLP(in_dim = config.hidden_size*2, out_dim = config.hidden_size)
        self.target_projector = projection_MLP(in_dim = config.hidden_size, out_dim = config.hidden_size)
        self.review_projector = projection_MLP(in_dim = config.hidden_size, out_dim = config.hidden_size)
        self.context_projector = projection_MLP(in_dim = config.hidden_size, out_dim = config.hidden_size)
        self.predictor = prediction_MLP(in_dim = config.hidden_size*2, hidden_dim = int(config.hidden_size / 2))
        
        # MLMヘッド
        # self.cls = BertOnlyMLMHead(config)
        self.cls = BertOnlyCombineMLMHead(config)
        
        # 結合用の層
        self.fusion_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        self.criterion = nn.CosineSimilarity(dim=1)
        self.init_weights()
        
    def forward(
        self,
        input_ids = None,
        labels=None,
        attention_mask=None,
        aug_input_ids=None,
        aug_labels= None,
        aug_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target_index = None,
        time_ids=None,
        aug_time_ids=None,
        traj_ids=None,
        timestamps=None,
        aug_timestamps=None,
        pois_triplet=None,
        user_ids=None,
    ):
        if input_ids[0, -1].item()==10000:
            stage='pretrain'
        else:
            stage='finetune'

        input_ids = input_ids[:, :-1].long()
        print(
        'inputids', input_ids,
        'labels', labels,
        attention_mask,
        aug_input_ids,
        aug_labels,
        aug_attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        target_index,
        time_ids,
        aug_time_ids,
        traj_ids,
        timestamps,
        aug_timestamps,
        pois_triplet,
        user_ids,)

        if stage == "pretrain":
            return self.forward_pretrain(
        input_ids =input_ids,
        labels=labels,
        attention_mask=attention_mask,
        aug_input_ids=aug_input_ids,
        aug_labels= aug_labels,
        aug_attention_mask=aug_attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        target_index = target_index,
        time_ids=time_ids,
        aug_time_ids=aug_time_ids,
        traj_ids=traj_ids,
        timestamps=timestamps,
        aug_timestamps=aug_timestamps,
        pois_triplet=pois_triplet,
        user_ids=user_ids,
            )
        else:
            return self.forward_finetune(
        input_ids =input_ids,
        labels=labels,
        attention_mask=attention_mask,
        aug_input_ids=aug_input_ids,
        aug_labels= aug_labels,
        aug_attention_mask=aug_attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        target_index = target_index,
        time_ids=time_ids,
        aug_time_ids=aug_time_ids,
        traj_ids=traj_ids,
        timestamps=timestamps,
        aug_timestamps=aug_timestamps,
        pois_triplet=pois_triplet,
        user_ids=user_ids,
            )
    def compute_contrastive_loss(self, anchor, positive, negatives, temperature=0.07):
        """
        InfoNCE Loss の実装
        
        Args:
            anchor: アンカーサンプル [batch_size, dim]
            positive: 正例サンプル [batch_size, dim]
            negatives: ネガティブサンプル [batch_size, dim]
            temperature: 温度パラメータ (デフォルト: 0.07)
        
        Returns:
            loss: InfoNCE Loss の値
        """
        batch_size = anchor.size(0)
        
        # 特徴量を正規化
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=1)
        
        # ポジティブペアの類似度
        pos_sim = torch.sum(anchor * positive, dim=-1) / temperature
        
        # ネガティブペアの類似度
        neg_sim = torch.mm(anchor, negatives.t()) / temperature  # [batch_size, batch_size]
        
        # 対角要素を除外（自分自身との類似度を除外）
        mask = torch.eye(batch_size, device=anchor.device)
        neg_sim = neg_sim.masked_fill(mask.bool(), float('-inf'))
        
        # すべての類似度を結合
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, batch_size + 1]
        
        # ラベルは常に0（最初の要素が正例）
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        return F.cross_entropy(logits, labels)
 
    def forward_finetune(
       self,
        input_ids = None,
        labels=None,
        attention_mask=None,
        aug_input_ids=None,
        aug_labels= None,
        aug_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target_index = None,
        time_ids=None,
        aug_time_ids=None,
        traj_ids=None,
        timestamps=None,
        aug_timestamps=None,
        pois_triplet=None,
        user_ids=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        z1_context_representation, z1_context_pooled = self.bert(
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
        )[:2]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(z1_context_representation.size()).float()
        sum_embeddings = torch.sum(z1_context_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z1_context_pooled = sum_embeddings / sum_mask

        z1_context = self.context_projector(z1_context_pooled)
        # p1_context = self.predictor(z1_context)
        # head1_context = self.cls(z1_context_representation)


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        z1_review_representation, z1_review_pooled = self.bert(
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
        )[:2]
        #print('z1 pooled', z1_pooled.shape)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(z1_review_representation.size()).float()
        sum_embeddings = torch.sum(z1_review_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z1_review_pooled = sum_embeddings / sum_mask

        z1_review= self.review_projector(z1_review_pooled)
        # p1_review = self.predictor(z1_review)
        # head1_review = self.cls(z1_review_representation)
        # print('z1', z1.shape, z1_pooled.shape)
        #print('cls z1', torch.cat([z1_context_representation, z1_review_representation], dim=2).shape)
        head1 = self.cls(torch.cat([z1_context_representation, z1_review_representation], dim=2))
        z1 = torch.cat([z1_context, z1_review],dim=1)
        #z1 = self.projector(z1)
        p1 = self.predictor(z1)

        if aug_input_ids != None:
            z2_context_representation, z2_context_pooled = self.bert(
                input_ids=aug_input_ids,
                attention_mask=aug_attention_mask,
                position_ids=position_ids,
                time_ids=aug_time_ids,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[:2]

            input_mask_expanded = aug_attention_mask.unsqueeze(-1).expand(z2_context_representation.size()).float()
            sum_embeddings = torch.sum(z2_context_representation * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min = 1e-9)
            z2_context_pooled = sum_embeddings / sum_mask

            z2_context = self.context_projector(z2_context_pooled)
            # p2_context = self.predictor(z2_context)
            # head2_context = self.cls(z2_context_representation)


            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            z2_review_representation, z2_review_pooled = self.bert(
                input_ids=aug_input_ids,
                attention_mask=aug_attention_mask,
                position_ids=position_ids,
                time_ids=aug_time_ids,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[:2]
            # print('z1 pooled', z2_pooled.shape)
            input_mask_expanded = aug_attention_mask.unsqueeze(-1).expand(z2_review_representation.size()).float()
            sum_embeddings = torch.sum(z2_review_representation * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min = 1e-9)
            z2_review_pooled = sum_embeddings / sum_mask

            z2_review= self.review_projector(z2_review_pooled)
            # p2_review = self.predictor(z2_review)
            # head2_review = self.cls(z2_review_representation)
            #print('z2', torch.cat([z2_context_representation, z2_review_representation], dim=1).shape)
            head2 = self.cls(torch.cat([z2_context_representation, z2_review_representation], dim=2))
            z2 = torch.cat([z2_context, z2_review],dim=1)
            p2 = self.predictor(z2)
            siamese_loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) / 2


        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            #print('labels', labels)
            #print('head1', head1.shape, head2.shape)
            masked_lm_loss_z1 = loss_fct(head1.view(-1, self.config.vocab_size), labels.view(-1))
            masked_lm_loss_z2 = loss_fct(head2.view(-1, self.config.vocab_size), aug_labels.view(-1))
            mask_loss = (masked_lm_loss_z1 + masked_lm_loss_z2) / 2
            #print('losses', mask_loss, siamese_loss)
            loss = mask_loss + siamese_loss
        else:
            loss = None

        return MaskedLMOutput(
            loss=loss,
            logits=head1,
        )
    
    def forward_pretrain(
       self,
        input_ids = None,
        labels=None,
        attention_mask=None,
        aug_input_ids=None,
        aug_labels= None,
        aug_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target_index = None,
        time_ids=None,
        aug_time_ids=None,
        traj_ids=None,
        timestamps=None,
        aug_timestamps=None,
        pois_triplet=None,
        user_ids=None,
    ):
        z_context_representation, z_context_pooled = self.bert(
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
        )[:2]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(z_context_representation.size()).float()
        sum_embeddings = torch.sum(z_context_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z_context_pooled = sum_embeddings / sum_mask
        # コンテキスト特徴量の取得
        # z_context = self.bert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask
        # ).last_hidden_state
        z_review_representation, z_review_pooled = self.review_bert(
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
        )[:2]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(z_review_representation.size()).float()
        sum_embeddings = torch.sum(z_review_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z_review_pooled = sum_embeddings / sum_mask
        # # レビュー特徴量の取得
        # z_review = self.review_bert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask
        # ).last_hidden_state
        z_target_representation, z_target_pooled = self.target_bert(
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
        )[:2]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(z_target_representation.size()).float()
        sum_embeddings = torch.sum(z_target_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z_target_pooled = sum_embeddings / sum_mask
        
        z_review_representation, z_review_pooled = self.review_bert(
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
        )[:2]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(z_review_representation.size()).float()
        sum_embeddings = torch.sum(z_review_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z_review_pooled = sum_embeddings / sum_mask
        
        # 特徴量の投影
        z_context_proj = self.context_projector(z_context_pooled)
        z_target_proj = self.target_projector(z_target_pooled)
        z_review_proj = self.review_projector(z_review_pooled)

        # z_context_proj = self.projector(z_context.mean(dim=1))
        # z_review_proj = self.projector(z_review.mean(dim=1))
        # z_target_proj = self.projector(z_target.mean(dim=1))
        
        # Contrastive Loss の計算
        loss_context = self.compute_contrastive_loss(z_target_proj, z_context_proj, z_context_proj)
        loss_review = self.compute_contrastive_loss(z_target_proj, z_review_proj, z_review_proj)
        
        loss = loss_context + loss_review
        return {
            'loss': loss, #loss_context + loss_review,
            'z_context': z_context_proj,
            'z_review': z_review_proj,
            'z_target': z_target_proj
        }
    
class SiamBERT(BertForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.config = config
        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.bert = BertModel(config, add_pooling_layer=True)
        self.projector = projection_MLP(in_dim = config.hidden_size, out_dim = config.hidden_size)
        self.predictor = prediction_MLP(in_dim = config.hidden_size, hidden_dim = int(config.hidden_size / 2))
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids = None,
        labels=None,
        attention_mask=None,
        aug_input_ids=None,
        aug_labels= None,
        aug_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target_index = None,
        time_ids=None,
        aug_time_ids=None,
        traj_ids=None,
        timestamps=None,
        aug_timestamps=None,
        pois_triplet=None,
        user_ids=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        z1_representation, z1_pooled = self.bert(
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
        )[:2]
        print('z1 pooled', z1_pooled.shape)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(z1_representation.size()).float()
        sum_embeddings = torch.sum(z1_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z1_pooled = sum_embeddings / sum_mask

        z1 = self.projector(z1_pooled)
        p1 = self.predictor(z1)
        head1 = self.cls(z1_representation)
        print('z1', z1.shape, z1_pooled.shape)

        if aug_input_ids != None:
            z2_representation, z2_pooled = self.bert(
                input_ids=aug_input_ids,
                attention_mask=aug_attention_mask,
                position_ids=position_ids,
                time_ids=aug_time_ids,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[:2]
            print('z2 pooled', z2_pooled.shape)
            input_mask_expanded = aug_attention_mask.unsqueeze(-1).expand(z2_representation.size()).float()
            sum_embeddings = torch.sum(z2_representation * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min = 1e-9)
            z2_pooled = sum_embeddings / sum_mask
            z2 = self.projector(z2_pooled)
            p2 = self.predictor(z2)
            head2 = self.cls(z2_representation)
            siamese_loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) / 2

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            print('labels', labels)
            print('head1', head1.shape, head2.shape)
            masked_lm_loss_z1 = loss_fct(head1.view(-1, self.config.vocab_size), labels.view(-1))
            masked_lm_loss_z2 = loss_fct(head2.view(-1, self.config.vocab_size), aug_labels.view(-1))
            mask_loss = (masked_lm_loss_z1 + masked_lm_loss_z2) / 2
            print('losses', mask_loss, siamese_loss)
            loss = mask_loss + siamese_loss
        else:
            loss = None

        return MaskedLMOutput(
            loss=loss,
            logits=head1,
        )
