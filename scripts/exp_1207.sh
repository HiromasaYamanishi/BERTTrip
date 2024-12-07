(cd BERT_Trip && CUDA_VISIBLE_DEVICES=0 python run.py --dataset ishikawa3) &
(cd BERT_Trip && CUDA_VISIBLE_DEVICES=1 python run.py --dataset kyoto3) &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=2 python run.py --dataset ishikawa3 --pretrain_epochs 0) &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=3 python run.py --dataset ishikawa3 --pretrain_epochs 10) &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=4 python run.py --dataset ishikawa3 --pretrain_epochs 20) &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=5 python run.py --dataset kyoto3 --pretrain_epochs 0) &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=6 python run.py --dataset kyoto3 --pretrain_epochs 10) &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=7 python run.py --dataset kyoto3 --pretrain_epochs 20) &
wait
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=0 python run.py --dataset ishikawa3 --pretrain_epochs 50)  &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=1 python run.py --dataset ishikawa3 --pretrain_epochs 20) --pretrain_batch_size 64 &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=2 python run.py --dataset ishikawa3 --pretrain_epochs 20) --pretrain_batch_size 32 &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=3 python run.py --dataset ishikawa3 --pretrain_epochs 20) --pretrain_batch_size 256 &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=4 python run.py --dataset kyoto3 --pretrain_epochs 50) &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=5 python run.py --dataset kyoto3 --pretrain_epochs 20) --pretrain_batch_size 64 &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=6 python run.py --dataset kyoto3 --pretrain_epochs 20) --pretrain_batch_size 32 &
(cd BERT_Trip_Review && CUDA_VISIBLE_DEVICES=7 python run.py --dataset kyoto3 --pretrain_epochs 20) --pretrain_batch_size 256 &
wait