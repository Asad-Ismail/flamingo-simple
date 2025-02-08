import os
import torch
from transformers import get_linear_schedule_with_warmup
from src.factory import create_model_and_transforms
import sys
from einops import rearrange
from train.data import get_data  # This will use the existing implementation

class TrainingConfig:
    def __init__(self):
        # Model paths
        self.vision_encoder_path = "ViT-L-14"
        self.vision_encoder_pretrained = "openai"
        self.lm_path = "facebook/opt-125m"  # Using smallest OPT model
        self.tokenizer_path = "facebook/opt-125m"
        
        # Training settings
        self.batch_size_mmc4 = 2
        self.batch_size_laion = 2
        self.train_num_samples_mmc4 = 30
        self.train_num_samples_laion = 50
        self.num_epochs = 1
        self.warmup_steps = 10
        self.learning_rate = 1e-4
        self.precision = "fp32"
        
        # Data paths
        self.laion_shards = "./minimal_datasets/laion-shard-{0000..0000}.tar"
        self.mmc4_shards = "./minimal_datasets/mmc4-shard-{0000..0000}.tar"
        
        # MMC4 specific settings
        self.mmc4_min_num_images = 1
        self.mmc4_max_num_images = 2
        self.mmc4_max_num_tokens = 256
        self.mmc4_textsim_threshold = 0.24
        
        # LAION specific settings
        self.laion_max_num_tokens = 32
        self.loss_multiplier_laion = 0.2
        
        # Distributed training settings
        self.workers = 1
        self.world_size = 1
        self.rank = 0
        self.distributed = False
        self.local_rank = 0
        self.device_id = 0
        
        # Dataset settings
        self.dataset_resampled = True
        self.seed = 42
        self.shuffle_buffer_size = 1000
        self.shuffle_seed = 42
        
        # Other settings
        self.cross_attn_every_n_layers = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gradient_checkpointing = True
        self.precision = "fp32"
        self.report_to_wandb = False
        
        # Tokenizer settings
        self.tokenizer_model_max_length = 2048
        
        # Logging settings
        self.log_steps = 5
        self.save_steps = 100
        
        # Debugging
        self.debug = True  # Set to True to enable more verbose output

def setup_model(config):
    """Initialize the model, tokenizer, and image processor"""
    print("Setting up model...")
    
    # Create model and transforms
    print("Creating model and transforms...")
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=config.vision_encoder_path,
        clip_vision_encoder_pretrained=config.vision_encoder_pretrained,
        lang_encoder_path=config.lm_path,
        tokenizer_path=config.tokenizer_path,
        cross_attn_every_n_layers=config.cross_attn_every_n_layers,
    )
      
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    # Move model to device
    print(f"Moving model to {config.device}")
    model = model.to(config.device)
    
    return model, tokenizer, image_processor

def setup_dataloaders(config, image_processor, tokenizer):
    """Setup data loaders using existing OpenFlamingo implementation"""
    print("Setting up dataloaders...")
    
    # Using the existing get_data function from OpenFlamingo
    laion_dataset = get_data(config, image_processor, tokenizer, "image_text")
    mmc4_dataset = get_data(config, image_processor, tokenizer, "mmc4")
    
    return laion_dataset, mmc4_dataset

def train_step(model, batch_laion, batch_mmc4, optimizer, config, tokenizer):
    """Perform a single training step"""

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    #vision_x_laion, text_laion = batch_laion
    #### LAION FORWARD PASS ####
    images = batch_laion[0].to(config.device)
    images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
    input_ids = batch_laion[1][0].to(config.device)
    attention_mask = batch_laion[1][1].to(config.device)
    print(attention_mask)
    # set up labels; language model is expected to handle shifting
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[labels == media_token_id] = -100
    labels = labels.to(config.device)

    # Process LAION batch
    laion_loss = model(
        vision_x=images,
        lang_x=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    ).loss


    images = batch_mmc4[0].to(config.device)
    images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
    input_ids = torch.stack([x[0] for x in batch_mmc4[1]]).squeeze(1)
    attention_mask = torch.stack([x[1] for x in batch_mmc4[1]]).squeeze(1)

    # set up labels; language model is expected to handle shifting
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    for i in range(labels.shape[0]):
        # remove loss for any token before the first <image> token
        label_idx = 0
        while (
            label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
        ):
            labels[i][label_idx] = -100
            label_idx += 1

        # get index of all endofchunk tokens in the sequence
        endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
        for endofchunk_idx in endofchunk_idxs:
            token_idx = endofchunk_idx + 1
            while (
                token_idx < labels.shape[1]
                and labels[i][token_idx] != media_token_id
            ):
                labels[i][token_idx] = -100
                token_idx += 1

    labels[labels == media_token_id] = -100
    labels = labels.to(config.device)

    # Process MMC4 batch
    mmc4_loss = model(
        vision_x=images,
        lang_x=input_ids.to(config.device),
        attention_mask=attention_mask.to(config.device),
        labels=labels,
    )[0]
    # Combined loss
    loss = laion_loss + mmc4_loss
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item(), laion_loss.item(), mmc4_loss.item()

def main():
    # Initialize config
    config = TrainingConfig()
    
    # Setup model and tokenizer
    model, tokenizer, image_processor = setup_model(config)
    
    # Setup dataloaders
    laion_dataset, mmc4_dataset = setup_dataloaders(config, image_processor, tokenizer)

    
    # Calculate total training steps
    total_training_steps = (config.train_num_samples_mmc4 // config.batch_size_mmc4) * config.num_epochs
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_training_steps
    )
    
    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Set epoch for both datasets
        laion_dataset.set_epoch(epoch)
        mmc4_dataset.set_epoch(epoch)
        
        # Get dataloaders
        laion_loader = laion_dataset.dataloader
        mmc4_loader = mmc4_dataset.dataloader

        # Get iterators 
        laion_iter = iter(laion_loader)
        mmc4_iter = iter(mmc4_loader)

        step = 0
        while step < min(config.train_num_samples_mmc4, config.train_num_samples_laion):
            try:
                # Get batches independently
                batch_laion = next(laion_iter)
                batch_mmc4 = next(mmc4_iter)
                
                # Training step
                loss, laion_loss, mmc4_loss = train_step(model, batch_laion, batch_mmc4, optimizer, config, tokenizer)
                
                # Update learning rate
                lr_scheduler.step()
                
                if step % config.log_steps == 0:
                    print(f"Step {step}: Loss = {loss:.4f} (LAION: {laion_loss:.4f}, MMC4: {mmc4_loss:.4f})")
                
                step += 1
                
            except StopIteration:
                # If either dataset runs out, reset both iterators
                print("Resetting dataset iterators...")
                laion_iter = iter(laion_loader)
                mmc4_iter = iter(mmc4_loader)
        
        # Save checkpoint
        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }, f'checkpoint_epoch_{epoch}.pt')

if __name__ == "__main__":
    main()