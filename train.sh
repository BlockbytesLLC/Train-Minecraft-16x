export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="James-A/Minecraft-16x-Dataset"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=128 --center_crop \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=2e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-minecraft-16x" \
  --dataloader_num_workers=8 \
  --report_to="wandb" \
  --num_processes=0 \
  --num_machines=1 \
  --mixed_precision="no" \
  --dynamo_backend="no"