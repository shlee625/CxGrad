#!/bin/sh
cd ..
export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$1

if [ -z $1 ]
then
	echo "Please specify the GPU_ID to use."
	exit 1
fi

python train_maml_system.py \
		--name_of_args_json_file experiment_config/tiered_imagenet_5way_5shot/CxGrad_4conv.json