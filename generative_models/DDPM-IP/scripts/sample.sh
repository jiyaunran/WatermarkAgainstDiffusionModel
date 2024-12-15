#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -m parameterA -i parameterB -b parameterC -n parameterD -o parameterE"
   echo "-m              model directory"
   echo "-i              image resolution size"
   echo "-b              batch size"
   echo "-n              number of samples"
   echo "-o              output directory"
   echo "-c				 cuda device"
   exit 1 # Exit script after printing help
}

model_dir=""
image_resolution=""
batch_size=""
num_samples=""
output_dir=""

while getopts "m:i:b:n:o:c:" opt; do
   case "$opt" in
      m ) model_dir="$OPTARG" ;;
      i ) image_resolution="$OPTARG" ;;
      b ) batch_size="$OPTARG" ;;
      n ) num_samples="$OPTARG" ;;
      o ) output_dir="$OPTARG" ;;
	  c ) cuda="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Check if required parameters are provided
if [ -z "$model_dir" ] || [ -z "$image_resolution" ] || [ -z "$batch_size" ] || [ -z "$num_samples" ] || [ -z "$output_dir" ]; then
    echo "Error: Missing required parameters."
    helpFunction
fi

python image_sample.py \
--model_path "$model_dir" \
--image_size "$image_resolution" \
--use_fp16 True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --timestep_respacing 100 \
--attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True \
--batch_size "$batch_size" \
--num_samples "$num_samples" \
--output_dir "${output_dir}" \
--cuda "$cuda"