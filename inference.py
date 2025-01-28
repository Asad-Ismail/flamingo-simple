import os
import torch
from PIL import Image
import requests
from src.factory import create_model_and_transforms
from huggingface_hub import hf_hub_download

class InferenceConfig:
    def __init__(self):
        # Model paths
        self.vision_encoder_path = "ViT-L-14"
        self.vision_encoder_pretrained = "openai"
        self.lm_path = "anas-awadalla/mpt-1b-redpajama-200b"
        self.tokenizer_path = "anas-awadalla/mpt-1b-redpajama-200b"
        self.checkpoint_path = None  # Will be downloaded from HF
        
        # Generation settings
        self.max_new_tokens = 20
        self.num_beams = 3
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Changing this to cpu if OOM error
        self.device = "cpu"

def load_model(config):
    """Initialize the model, tokenizer, and image processor"""
    print("Setting up model...")

    # Create model and transforms
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=config.vision_encoder_path,
        clip_vision_encoder_pretrained=config.vision_encoder_pretrained,
        lang_encoder_path=config.lm_path,
        tokenizer_path=config.tokenizer_path,
        cross_attn_every_n_layers=1
    )
    # Download and load checkpoint from HuggingFace
    checkpoint_path = hf_hub_download(
        "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        "checkpoint.pt",
        local_dir="openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        local_dir_use_symlinks=False
    )
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    
    # Move model to device and set to eval mode
    model = model.to(config.device)
    model.eval()
    
    # Set tokenizer padding side
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    
    return model, image_processor, tokenizer

def load_demo_images():
    """Load example images from COCO dataset"""
    print("Loading demo images...")
    
    # Load three demo images
    demo_image_one = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", 
            stream=True
        ).raw
    )

    demo_image_two = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
            stream=True
        ).raw
    )

    query_image = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
            stream=True
        ).raw
    )
    
    return [demo_image_one, demo_image_two, query_image]

def prepare_images(images, image_processor, device):
    """Process a list of PIL images into tensor format expected by OpenFlamingo
    
    Returns: Tensor of shape (batch_size, num_media, num_frames, channels, height, width)
    """
    # Process each image and create a list of tensors
    vision_x = [image_processor(img).unsqueeze(0) for img in images]
    
    # Stack along batch dimension
    vision_x = torch.cat(vision_x, dim=0)
    
    # Add num_media and num_frames dimensions
    # Shape: (batch_size=1, num_media=3, num_frames=1, channels=3, height=224, width=224)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    
    return vision_x.to(device)

def prepare_text_prompt(tokenizer, device):
    """Prepare text prompt with special tokens for OpenFlamingo"""
    # Add special tokens for each image:
    # <image> for image location
    # <|endofchunk|> for end of text associated with an image
    prompt = [
        "<image>An image of two cats.<|endofchunk|>"
        "<image>An image of a bathroom sink.<|endofchunk|>"
        "<image>An image of"
    ]
    
    # Tokenize with left-side padding for generation
    lang_x = tokenizer(prompt, return_tensors="pt")
    
    # Move to device
    lang_x = {
        "input_ids": lang_x["input_ids"].to(device),
        "attention_mask": lang_x["attention_mask"].to(device)
    }
    
    return lang_x

def generate_response(model, vision_x, lang_x, config):
    """Generate text response given images and text prompt"""
    with torch.no_grad():
        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
        )
    return generated_text

def main():
    # Initialize config
    config = InferenceConfig()
    
    # Load model, tokenizer, and image processor
    model, image_processor, tokenizer = load_model(config)
    
    # Load demo images
    images = load_demo_images()
    
    # Process images into correct format
    vision_x = prepare_images(images, image_processor, config.device)
    
    # Prepare text prompt
    lang_x = prepare_text_prompt(tokenizer, config.device)
    
    # Generate completion
    print("\nGenerating response...")
    generated_text = generate_response(model, vision_x, lang_x, config)
    
    # Decode and print result
    print("\nGenerated text:", tokenizer.decode(generated_text[0]))

if __name__ == "__main__":
    main() 