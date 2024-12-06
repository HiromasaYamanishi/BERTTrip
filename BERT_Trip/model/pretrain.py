import torch
import os.path
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from typing import Dict
from transformers import PreTrainedTokenizer, BertTokenizer, BertConfig, TrainingArguments, Trainer
from .data_collator import  DataCollatorForLanguageModeling
from .bert.bert_model import BertTripConfig
from datetime import datetime
from util import datetime_to_interval

class TripTrainer:
    def __init__(self, model, **kwargs):
        print('trip planner kwargs', kwargs)
        config = BertTripConfig(
            **kwargs,
        )
        #print('trip planner config', config)
        self.config = config
        #print('self config', self.config)
        self.base_model = model
        self.model = self.base_model(config)
        if os.path.isfile(f'{config.pretrained_model_dir}/config.json'):
            self.model = self.model.from_pretrained(config.pretrained_model_dir)
        self.model_dir = config.pretrained_model_dir
        print('No of parameters: ', self.model.num_parameters(), config.hidden_size)

        self.poi_tokenizer = BertTokenizer(vocab_file = config.poi_vocab_file, do_lower_case = False, do_basic_tokenize = False)

        self.data_collator = DataCollatorForLanguageModeling(
            dataset = config.dataset,
            tokenizer = self.poi_tokenizer,
            mlm = True,
            mlm_probability = config.mlm_probability,
            config = config,
        )

    def reset(self):
        self.model = self.base_model(self.config)

    def train(self, dataset_file_path = None, epochs = 50, batch_size = 32, save_steps = 5000, save_model = True):
        if dataset_file_path == None:
            dataset_file_path = self.config.train_data
        max_sequence_length = self.config.max_position_embeddings
        #print('dataset_file_path', dataset_file_path)
        dataset = LineByLineTextDataset(
            tokenizer = self.poi_tokenizer,
            file_path = dataset_file_path,
            block_size = max_sequence_length,
            config = self.model.config,
        )
        training_args = TrainingArguments(
            output_dir='./',
            overwrite_output_dir=True,
            num_train_epochs = epochs,
            per_device_train_batch_size = batch_size,
            save_steps = save_steps,
        )
        trainer = Trainer(
            model = self.model,
            args = training_args,
            data_collator = self.data_collator,
            train_dataset = dataset,
        )
        # print('trainer', trainer)
        # print('model', self.model)
        # print('args', training_args)
        # print('collator', self.data_collator)
        # print('dataset', dataset)
        # exit()
        trainer.train()
        if save_model:
            trainer.save_model(self.model_dir)
            

class TripReviewTrainer:
    def __init__(self, model, **kwargs):
        print('trip planner kwargs', kwargs)
        self.config = BertTripConfig(**kwargs)
        self.base_model = model
        self.model = self.base_model(self.config)
        
        if os.path.isfile(f'{self.config.pretrained_model_dir}/config.json'):
            self.model = self.model.from_pretrained(self.config.pretrained_model_dir)
        
        self.model_dir = self.config.pretrained_model_dir
        print('No of parameters: ', self.model.num_parameters(), self.config.hidden_size)

        self.poi_tokenizer = BertTokenizer(
            vocab_file=self.config.poi_vocab_file,
            do_lower_case=False,
            do_basic_tokenize=False
        )

    def reset(self):
        self.model = self.base_model(self.config)

    def train(self, dataset_file_path=None, epochs=50, batch_size=32, save_steps=5000, save_model=True):
        """二段階学習のメインメソッド"""
        if dataset_file_path is None:
            dataset_file_path = self.config.train_data

        # Step 1: Contrastive Pretraining
        print("Starting contrastive pretraining...")
        self._contrastive_pretrain(
            dataset_file_path,
            epochs=self.config.pretrain_epochs,
            batch_size=batch_size
        )

        # Step 2: Fine-tuning
        print("Starting fine-tuning...")
        self._fine_tune(
            dataset_file_path,
            epochs=epochs,
            batch_size=batch_size,
            save_steps=save_steps,
            save_model=save_model
        )

    def _contrastive_pretrain(self, dataset_file_path, epochs, batch_size):
        """Contrastive learningによるpretraining"""
        max_sequence_length = self.config.max_position_embeddings
        
        # Pretraining用のデータセットを作成
        pretrain_dataset = ReviewLineByLineDataset(
            tokenizer=self.poi_tokenizer,
            file_path=dataset_file_path,
            block_size=max_sequence_length,
            config=self.model.config
        )

        # Pretraining用のデータコレーターを作成
        pretrain_collator = ReviewDataCollatorForContrastiveLearning(
            tokenizer=self.poi_tokenizer,
            config=self.config
        )

        # Pretraining用のTraining Arguments
        pretrain_args = TrainingArguments(
            output_dir='./pretrain_checkpoint',
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=self.config.pretrain_lr,
            save_steps=1000,
            logging_steps=100,
        )

        # Pretraining用のTrainerを作成
        pretrain_trainer = ReviewPreTrainer(
            model=self.model,
            args=pretrain_args,
            data_collator=pretrain_collator,
            train_dataset=pretrain_dataset,
        )

        # Pretrainingの実行
        pretrain_trainer.train()

    def _fine_tune(self, dataset_file_path, epochs, batch_size, save_steps, save_model):
        """Fine-tuning stage"""
        max_sequence_length = self.config.max_position_embeddings
        
        # Fine-tuning用のデータセット
        dataset = LineByLineTextDataset(
            tokenizer=self.poi_tokenizer,
            file_path=dataset_file_path,
            block_size=max_sequence_length,
            config=self.model.config,
        )

        # 通常のMLMデータコレーター
        self.data_collator = DataCollatorForLanguageModeling(
            dataset=self.config.dataset,
            tokenizer=self.poi_tokenizer,
            mlm=True,
            mlm_probability=self.config.mlm_probability,
            config=self.config,
        )

        # Fine-tuning用のTraining Arguments
        training_args = TrainingArguments(
            output_dir='./',
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
        )

        # Fine-tuning用のTrainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=dataset,
        )

        # Fine-tuningの実行
        trainer.train()
        
        if save_model:
            trainer.save_model(self.model_dir)

class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self,
    tokenizer: PreTrainedTokenizer,
    file_path: str,
    block_size: int,
    config: BertTripConfig,
    sep = ','):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        self.config = config
        users = []
        trajectories = []
        aug_trajectories = []

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            for line in lines:
                features = line.split('|')
                user_id = features[0]
                trajectory_list = np.array(features[1].split(sep))
                times = np.array([datetime.fromtimestamp(int(i)) for i in features[-1].split(sep)])
                inputs = self.aug(user_id, trajectory_list, times)
                aug_inputs = self.aug(user_id, trajectory_list, times)
                trajectories.append(inputs)
                aug_trajectories.append(aug_inputs)

        batch_encoding = tokenizer(trajectories, add_special_tokens=True, truncation=True, max_length = block_size)
        aug_batch_encoding = tokenizer(aug_trajectories, add_special_tokens=True, truncation=True, max_length = block_size)

        size = len(batch_encoding['input_ids'])
        self.examples = [{
            "input_ids": torch.tensor(batch_encoding['input_ids'][i], dtype=torch.long),
            "aug_input_ids": torch.tensor(aug_batch_encoding['input_ids'][i], dtype=torch.long),
        } for i in range(size)]

    def aug(self, user, trajectory_list, times):
        length = len(trajectory_list)
        inputs = np.array([])

        if self.config.add_user_token:
            inputs = np.concatenate((inputs, [user]))

        if self.config.use_data_agumentation:
            new_length = np.random.randint(3, length + 1)
            choices = np.sort(np.random.choice(np.arange(length), new_length, replace = False))
            trajectory = trajectory_list[choices]
        else:
            choices = np.arange(length)
            trajectory = trajectory_list

        time_chosen = times[choices]
        time_chosen = [datetime_to_interval(t) for t in time_chosen]
        time_tokens = np.array([f'time-{_}' for _ in time_chosen])

        time_tokens = [time_tokens[0], time_tokens[-1]]
        if self.config.add_time_token:
            inputs = np.concatenate((inputs, time_tokens))

        inputs = np.concatenate((inputs, trajectory))
        inputs = ' '.join(inputs)

        return inputs
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
