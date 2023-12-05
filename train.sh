export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="James-A/Minecraft-16x-Dataset"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=128 --center_crop \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=10000 \
  --learning_rate=2e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=500 \
  --output_dir="sd-minecraft-16x" \
  --dataloader_num_workers=8 \
  --report_to="wandb"