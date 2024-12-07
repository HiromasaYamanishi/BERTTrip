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
    def __init__(self, model, config, **kwargs):
        print('trip planner kwargs', kwargs)
        config = config
        #print('trip planner config', config)
        self.config = config
        #print('self config', self.config)
        self.base_model = model
        self.model = self.base_model(config)
        #self.model = model
        if os.path.isfile(f'{config.pretrained_model_dir}/config.json'):
            self.model = self.model.from_pretrained(config.pretrained_model_dir)
        self.model_dir = config.pretrained_model_dir
        print('No of parameters: ', self.model.num_parameters(), config.hidden_size)

        self.poi_tokenizer = BertTokenizer(vocab_file = config.poi_vocab_file, do_lower_case = False, do_basic_tokenize = False)
        self.review_tokenizer = BertTokenizer(vocab_file = config.review_vocab_file, do_lower_case = False, do_basic_tokenize = False)
        self.data_collator = DataCollatorForLanguageModeling(
            dataset = config.dataset,
            poi_tokenizer = self.poi_tokenizer,
            review_tokenizer=self.review_tokenizer,
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
        dataset = LineByLineTextDataset(
            poi_tokenizer = self.poi_tokenizer,
            review_tokenizer=self.review_tokenizer,
            file_path = dataset_file_path,
            block_size = max_sequence_length,
            config = self.model.config,
        )
        # print('dataset 0', dataset[0])
        # print('dataset 1', dataset[1])
        # exit()
        training_args = TrainingArguments(
            output_dir='./',
            overwrite_output_dir=True,
            num_train_epochs = epochs,
            per_device_train_batch_size = batch_size,
            #remove_unused_columns=False,
            save_steps = save_steps,
        )
        trainer = Trainer(
            model = self.model,
            args = training_args,
            data_collator = self.data_collator,
            train_dataset = dataset,
        )
        trainer.train()
        if save_model:
            trainer.save_model(self.model_dir)
            

class ReviewPreTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Contrastive learningのための損失計算
        """
        outputs = model(
            inputs,
            stage="pretrain"
        )

        loss = outputs['loss']

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        学習ステップのカスタマイズ
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        print('inputs', inputs)
        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()
    
class TripReviewTrainer:
    def __init__(self, model, **kwargs):
        print('trip planner kwargs', kwargs)
        self.config = BertTripConfig(**kwargs)
        self.base_model = model
        self.model = self.base_model(self.config)
        print('base model', self.base_model)
        print('model', model)
        self.trip_trainer = TripTrainer(model, self.config, **kwargs)
        
        if os.path.isfile(f'{self.config.pretrained_model_dir}/config.json'):
            self.model = self.model.from_pretrained(self.config.pretrained_model_dir)
        
        self.init_kwargs = kwargs
        self.model_dir = self.config.pretrained_model_dir
        print('No of parameters: ', self.model.num_parameters(), self.config.hidden_size)

        self.poi_tokenizer = BertTokenizer(
            vocab_file=self.config.poi_vocab_file,
            do_lower_case=False,
            do_basic_tokenize=False
        )

    def reset(self):
        self.model = self.base_model(self.config)

    def train(self, dataset_file_path=None, epochs=50, batch_size=32, save_steps=5000, save_model=True, mode='contrastive'):
        """二段階学習のメインメソッド"""
        if dataset_file_path is None:
            dataset_file_path = self.config.train_data

        # Step 1: Contrastive Pretraining
        print("Starting contrastive pretraining...")
        if mode=='contrastive':
            self._contrastive_pretrain(
                dataset_file_path,
                epochs=epochs, #epochs,
                batch_size=batch_size,
                save_steps=save_steps,
                save_model=save_model
            )

        # Step 2: Fine-tuning
        elif mode =='finetune':
            print("Starting fine-tuning...")
            self._fine_tune(
                dataset_file_path,
                epochs=epochs, #epochs,
                batch_size=batch_size,
                save_steps=save_steps,
                save_model=save_model
            )

    def _contrastive_pretrain(self, dataset_file_path, epochs, batch_size, save_steps, save_model):
        """Contrastive learningによるpretraining"""
        max_sequence_length = self.config.max_position_embeddings
        
        # # Pretraining用のデータセットを作成
        # pretrain_dataset = ReviewLineByLineDataset(
        #     tokenizer=self.poi_tokenizer,
        #     file_path=dataset_file_path,
        #     block_size=max_sequence_length,
        #     config=self.model.config
        # )

        # # Pretraining用のデータコレーターを作成
        # pretrain_collator = ReviewDataCollatorForContrastiveLearning(
        #     tokenizer=self.poi_tokenizer,
        #     config=self.config
        # )

        # Pretraining用のTraining Arguments
        # pretrain_args = TrainingArguments(
        #     output_dir='./pretrain_checkpoint',
        #     overwrite_output_dir=True,
        #     num_train_epochs=epochs,
        #     per_device_train_batch_size=batch_size,
        #     learning_rate=self.config.pretrain_lr,
        #     save_steps=1000,
        #     logging_steps=100,
        # )
        self.trip_trainer.data_collator.stage = 'pretrain'
        self.trip_trainer.train(dataset_file_path = dataset_file_path, 
                      epochs = epochs, batch_size = batch_size, 
                      save_steps = save_steps, save_model = save_model)


    def _fine_tune(self, dataset_file_path, epochs, batch_size, save_steps, save_model):
        """Fine-tuning stage"""
        print('self trip trainer', self.trip_trainer)
        self.trip_trainer.data_collator.stage = 'finetune'
        self.trip_trainer.train(dataset_file_path = dataset_file_path, 
                      epochs = epochs, batch_size = batch_size, 
                      save_steps = save_steps, save_model = save_model)
        
        if save_model:
            self.trip_trainer.save_model(self.model_dir)

class LineByLineTextDataset(Dataset):
    def __init__(self, poi_tokenizer: PreTrainedTokenizer, review_tokenizer:PreTrainedTokenizer, file_path: str, block_size: int, config: BertTripConfig, sep=','):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        self.config = config
        
        trajectories = []
        aug_trajectories = []
        review_trajectories = []
        aug_review_trajectories = []

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            for line in lines:
                features = line.split('|')
                user_id = features[0]
                trajectory_list = np.array(features[1].split(sep))
                review_list = np.array(features[-1].split(sep))  # review_idは最後
                times = np.array([datetime.fromtimestamp(int(i)) for i in features[-2].split(sep)])  # timestampは最後から2番目
                
                # 共通のaugmentationパターンを生成
                length = len(trajectory_list)
                if self.config.use_data_agumentation:
                    new_length = np.random.randint(3, length + 1)
                    choices = np.sort(np.random.choice(np.arange(length), new_length, replace=False))
                    aug_choices = np.sort(np.random.choice(np.arange(length), new_length, replace=False))
                else:
                    choices = np.arange(length)
                    aug_choices = np.arange(length)
                #print('review list', review_list, trajectory_list)
                # POI系列とreview系列で同じchoicesを使用
                inputs = self.aug(user_id, trajectory_list, times, choices)
                aug_inputs = self.aug(user_id, trajectory_list, times, aug_choices)
                review_inputs = self.aug(user_id, review_list, times, choices)  # 同じchoices
                aug_review_inputs = self.aug(user_id, review_list, times, aug_choices)  # 同じaug_choices

                trajectories.append(inputs)
                aug_trajectories.append(aug_inputs)
                review_trajectories.append(review_inputs)
                aug_review_trajectories.append(aug_review_inputs)

        # トークナイズ
        batch_encoding = poi_tokenizer(trajectories, add_special_tokens=True, truncation=True, max_length=block_size)
        aug_batch_encoding = poi_tokenizer(aug_trajectories, add_special_tokens=True, truncation=True, max_length=block_size)
        review_batch_encoding = review_tokenizer(review_trajectories, add_special_tokens=True, truncation=True, max_length=block_size)
        aug_review_batch_encoding = review_tokenizer(aug_review_trajectories, add_special_tokens=True, truncation=True, max_length=block_size)

        self.examples = [{
            "input_ids": torch.tensor(batch_encoding['input_ids'][i], dtype=torch.long),
            "aug_input_ids": torch.tensor(aug_batch_encoding['input_ids'][i], dtype=torch.long),
            "review_ids": torch.tensor(review_batch_encoding['input_ids'][i], dtype=torch.long),
            "aug_review_ids": torch.tensor(aug_review_batch_encoding['input_ids'][i], dtype=torch.long),
        } for i in range(len(batch_encoding['input_ids']))]
        # print(self.examples[:5])
        # exit()

    def aug(self, user, sequence_list, times, choices):
        """
        Modified aug function that takes choices as parameter
        """
        inputs = np.array([])
        if self.config.add_user_token:
            inputs = np.concatenate((inputs, [user]))

        sequence = sequence_list[choices]
        time_chosen = times[choices]
        time_chosen = [datetime_to_interval(t) for t in time_chosen]
        time_tokens = np.array([f'time-{_}' for _ in time_chosen])

        time_tokens = [time_tokens[0], time_tokens[-1]]
        if self.config.add_time_token:
            inputs = np.concatenate((inputs, time_tokens))

        inputs = np.concatenate((inputs, sequence))
        inputs = ' '.join(inputs)

        return inputs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        #print('__getitem__', i, self.examples[i])
        return self.examples[i]
    
class LineByLineTextDataset_(Dataset):
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
