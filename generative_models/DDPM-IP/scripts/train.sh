#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -d parameterA -i parameterB -b parameterC -s parameterD -v parameterE -r parameterF -c parameterG"
   echo "-d              data directory"
   echo "-i              image resolution size"
   echo "-b              batch size"
   echo "-s              stop step"
   echo "-v              stop save directory"
   echo "-r              resume directory"
   echo "-c              cuda device"
   exit 1 # Exit script after printing help
}

data_dir=""
image_resolution=""
batch_size=""
stop_step=""
stop_save_dir=""
resume_dir=""
cuda=""

while getopts "d:i:b:s:v:r:c:" opt; do
   case "$opt" in
      d ) data_dir="$OPTARG" ;;
      i ) image_resolution="$OPTARG" ;;
      b ) batch_size="$OPTARG" ;;
      s ) stop_step="$OPTARG" ;;
      v ) stop_save_dir="$OPTARG" ;;
      r ) resume_dir="$OPTARG" ;;
      c ) cuda="$OPTARG" ;;
      ? ) helpFunction ;; # Print help if parameter is non-existent
   esac
done

# Check if image resolution is set
if [ -z "$image_resolution" ]; then
    echo "Image resolution not set. Exiting..."
    helpFunction
fi

# Run commands based on image resolution
if [ "$image_resolution" -eq 32 ]; then
    python image_train.py --input_pertub 0.1 \
    --data_dir "$data_dir" \
    --image_size "$image_resolution" \
    --use_fp16 True --num_channels 128 --num_head_channels 32 --num_res_blocks 3 \
    --attention_resolutions 16,8 --resblock_updown True --use_new_attention_order True \
    --learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
    --rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 \
    --batch_size "$batch_size" \
    --stop_step "$stop_step" \
    --stop_save_dir "$stop_save_dir" \
    --resume_checkpoint "$resume_dir" \
    --cuda "$cuda"
elif [ "$image_resolution" -eq 64 ]; then
    python image_train.py --input_pertub 0.1 \
    --data_dir "$data_dir" \
    --image_size "$image_resolution" \
    --use_fp16 True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
    --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
    --learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
    --rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 \
    --batch_size "$batch_size" \
    --stop_step "$stop_step" \
    --stop_save_dir "$stop_save_dir" \
    --resume_checkpoint "$resume_dir" \
    --cuda "$cuda"
else
    python image_train.py --input_pertub 0.1 \
    --data_dir "$data_dir" \
    --image_size "$image_resolution" \
    --use_fp16 True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
    --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
    --learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
    --rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 \
    --batch_size "$batch_size" \
    --stop_step "$stop_step" \
    --stop_save_dir "$stop_save_dir" \
    --resume_checkpoint "$resume_dir" \
    --cuda "$cuda"
fi
