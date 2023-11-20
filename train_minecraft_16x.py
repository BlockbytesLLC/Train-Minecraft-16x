import os
import math
import torch
import wandb
import inspect
import torch.nn.functional as F
import matplotlib.pyplot as plt

from accelerate import Accelerator
from accelerate import notebook_launcher
from dataclasses import dataclass
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage, InterpolationMode
from tqdm.auto import tqdm

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid


def transform(examples):
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size), interpolation=InterpolationMode.NEAREST_EXACT),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),  # Updated for 4 channels
    ])
    images = [preprocess(image.convert("RGBA")) for image in examples["image"]]  # Convert to "RGBA"
    return {"images": images}


def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(-1, 1, 1)  # Updated for 4 channels
    std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(-1, 1, 1)   # Updated for 4 channels
    return tensor.mul(std).add(mean).clamp(0, 1)


def show_dataset_images(dataset):
    """
    Display a grid of images from the dataset.
    """
    num_images = 256
    grid_size = int(math.ceil(math.sqrt(num_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= num_images:
            ax.axis('off')
            continue
        data = dataset[idx]
        img_tensor = data['images']
        img_tensor = denormalize(img_tensor)  # Denormalize the image tensor
        img = ToPILImage()(img_tensor)

        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    sqr = int(math.sqrt(config.eval_batch_size))
    image_grid = make_image_grid(images, rows=sqr, cols=sqr)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    file_path = f"{test_dir}/{epoch:04d}.png"
    image_grid.save(file_path)
    return file_path

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # logs
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}

            # Log metrics to wandb
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                image_path = evaluate(config, epoch, pipeline)
                accelerator.log({"generated_images": wandb.Image(image_path, caption="Epoch: {}".format(epoch))}, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)


def props(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr


@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 64
    eval_batch_size = 36  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "minecraft-16x"  # the model name locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = "James-A/Minecraft-16x-Diffusion"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


# Initialise training config
config = TrainingConfig()

# Initialize a new wandb run
os.environ['WANDB_API_KEY'] = '0f0dcec292313063373073bfe861e02a31990e43'
wandb.init(project="train-minecraft-16x", entity="james-a", config=props(config))

# Load dataset
dataset = load_dataset("James-A/Minecraft-16x", split="train", drop_labels=False)

# Preprocess images in dataset
dataset.set_transform(transform)

# Visualise dataset
# show_dataset_images(dataset)

# Create model
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=4,  # the number of input channels
    out_channels=4,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(256, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
)

sample_image = dataset[0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=0).sample.shape)

# Sample image modifications
sample_image = dataset[0]["images"].unsqueeze(0)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

# Start training
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)

