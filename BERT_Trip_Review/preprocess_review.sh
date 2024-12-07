CUDA_VISIBLE_DEVICES=0 python preprocess_review.py --dataset tokyo3 &
CUDA_VISIBLE_DEVICES=1 python preprocess_review.py --dataset osaka3 &
CUDA_VISIBLE_DEVICES=2 python preprocess_review.py --dataset ishikawa3 &
CUDA_VISIBLE_DEVICES=3 python preprocess_review.py --dataset okinawa3 &
CUDA_VISIBLE_DEVICES=4 python preprocess_review.py --dataset hokkaido3 &
CUDA_VISIBLE_DEVICES=5 python preprocess_review.py --dataset kyoto3 &
wait