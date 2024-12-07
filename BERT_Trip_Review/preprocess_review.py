#from model.bert.sentence_bert import AspectCategoryClassifier
import pandas as pd
import shutil
import torch
import os
import argparse
import torch
import torch.nn as nn
from transformers import (
    BertJapaneseTokenizer, 
    BertModel
)
from sklearn.metrics import f1_score
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import argparse

class AspectCategoryClassifier(nn.Module):
    def __init__(self, model_name_or_path, num_aspects=6, num_categories=32, device='cuda'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)
        self.device = device
        
        # 埋め込みサイズ（BERTの出力次元）
        hidden_size = self.bert.config.hidden_size
        
        # アスペクトごとの分類ヘッド（0-3の4クラス分類）
        print('hidden aspect', hidden_size, num_aspects, num_categories)
        self.aspect_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 4) for _ in range(num_aspects)
        ])
        
        # POIカテゴリ分類ヘッド
        self.category_classifier = nn.Linear(hidden_size, num_categories)
        
        # 指定した数の層をフリーズ
        num_trainable_layers = 2
        self.freeze_except_last_n_layers(num_trainable_layers=num_trainable_layers)

    def freeze_except_last_n_layers(self, num_trainable_layers):
        """最後のN層以外をフリーズする"""
        # まず全層をフリーズ
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # BERTの全層数を取得
        total_layers = len(self.bert.encoder.layer)
        
        # 最後のN層のパラメータを学習可能にする
        for i in range(total_layers - num_trainable_layers, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
                
        # 分類層は常に学習可能
        for param in self.aspect_classifiers.parameters():
            param.requires_grad = True
        for param in self.category_classifier.parameters():
            param.requires_grad = True
    
    def _mean_pooling(self, model_output, attention_mask):
        """文章全体のmean poolingを計算"""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, input_ids, attention_mask, return_emb=False):
        # BERTの出力を取得
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # mean poolingを適用
        pooled_output = self._mean_pooling(outputs, attention_mask)
        
        pooled_output = self.dropout(pooled_output)
        
        # アスペクトごとの分類
        aspect_logits = [
            classifier(pooled_output) for classifier in self.aspect_classifiers
        ]
        
        # カテゴリ分類
        category_logits = self.category_classifier(pooled_output)
        
        if return_emb:
            return aspect_logits, category_logits, pooled_output
        return aspect_logits, category_logits
    
    @torch.no_grad()
    def encode(self, sentences, batch_size=200):
        tokenizer = BertJapaneseTokenizer.from_pretrained(
            "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
        )
        self.tokenizer = tokenizer
        all_embeddings = []
        iterator = tqdm(range(0, len(sentences), batch_size))
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.bert(input_ids=encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        if len(all_embeddings):
            return torch.stack(all_embeddings)
        else:
            return torch.zeros((0, 768))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, )
    args = parser.parse_args()
    dataset = args.dataset
    #for dataset in ['tokyo3', 'osaka3', 'kyoto3', 'ishikawa3', 'hokkaido3', 'okinawa3']:
    review_df = pd.read_csv(f'./data/{dataset}/review.csv')
    for fold in range(5):
        os.makedirs(f'./data/{dataset}/fold_{fold}/', exist_ok=True)
        shutil.copy(f'/home/yamanishi/project/airport/src/analysis/route_recommendation/cache/sbert_finetune/{dataset}/fold_{fold}/poi_embedding_cache.pt',
                    f'./data/{dataset}/fold_{fold}/poi_review_emb.pt')
    

        checkpoint_path = f'./checkpoint/{dataset}/best_model_fold{fold}.pt'
        checkpoint = torch.load(checkpoint_path)
        num_categories = checkpoint['category_classifier.weight'].size(0)
        print(f"Detected {num_categories} categories from checkpoint")
        model = AspectCategoryClassifier(
            "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
            num_aspects=11,
            num_categories=num_categories,
            device='cuda'
        )
        model.load_state_dict(checkpoint)
        model.to('cuda')
        embs = model.encode(review_df['review'].values)
        emb_d = {id:emb for id,emb in zip(review_df['reviewID'], embs)}
        torch.save(emb_d, f'./data/{dataset}/fold_{fold}/review_emb.pt')