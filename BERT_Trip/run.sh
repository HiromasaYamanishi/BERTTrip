eval "$(conda shell.bash hook)"
conda activate llava
cities=("osaka3" "okinawa3" "kyoto3" "hokkaido3" "tokyo3" "ishikawa3")
#cities=("tokyo")
gpus=("2" "4" "5" "7")

for i in "${!cities[@]}"; do
    city=${cities[$i]}
    #python preprocess.py --city ${city} &
    python ./data/preprocessor/preprocess.py --dataset ${city}
    python ./data/preprocessor/p2.py --dataset ${city}
done
wait
echo 'preprocess end'