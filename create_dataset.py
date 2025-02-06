import os
import json
import requests
import tarfile
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import random

def download_coco_annotations():
    """Download COCO captions annotations if not present"""
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    if not os.path.exists('annotations'):
        print("Downloading COCO annotations...")
        response = requests.get(url)
        with open('annotations.zip', 'wb') as f:
            f.write(response.content)
        import zipfile
        with zipfile.ZipFile('annotations.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove('annotations.zip')

def download_coco_image(image_id):
    """Download a single COCO image"""
    url = f"http://images.cocodataset.org/train2017/{image_id:012d}.jpg"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to download image {image_id}")
            return None
    except Exception as e:
        print(f"Error downloading image {image_id}: {e}")
        return None

def create_mmc4_sample(images, captions):
    """Create a single MMC4 sample with proper format"""
    data = {
        "text_list": captions,
        "image_info": [],
        "similarity_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "interleaved": False,
        "language": "en"
    }
    
    # Convert images to base64
    for img in images:
        img_buffer = BytesIO()
        img.save(img_buffer, format="JPEG")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        data["image_info"].append({
            "image_base64": img_base64,
            "image_height": img.height,
            "image_width": img.width
        })
    
    return data

def create_minimal_coco_dataset(output_dir, num_samples_laion=50, num_samples_mmc4=30):
    os.makedirs(output_dir, exist_ok=True)
    
    # Download and load COCO annotations
    download_coco_annotations()
    with open('annotations/captions_train2017.json', 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to captions mapping
    image_captions = {}
    for ann in coco_data['annotations']:
        if ann['image_id'] not in image_captions:
            image_captions[ann['image_id']] = []
        image_captions[ann['image_id']].append(ann['caption'])
    
    # Get all image IDs that have captions
    image_ids = list(image_captions.keys())
    random.shuffle(image_ids)
    
    # Split images between LAION and MMC4
    num_samples_laion = (num_samples_laion // 2) * 2
    print(f"Number of LAION samples: {num_samples_laion}")
    laion_images = image_ids[:num_samples_laion]
    mmc4_images = image_ids[num_samples_laion:num_samples_laion + num_samples_mmc4 * 2]
    
    # Create LAION-style dataset
    print("\nCreating LAION-style dataset...")
    with tarfile.open(os.path.join(output_dir, "laion-shard-0000.tar"), "w") as tar:
        for i, image_id in enumerate(tqdm(laion_images)):
            sample_id = f"{i:08d}"
            img = download_coco_image(image_id)
            if img is None:
                continue
            
            # Save image
            img_buffer = BytesIO()
            img.save(img_buffer, format="JPEG")
            img_bytes = img_buffer.getvalue()
            
            # Create WebDataset compatible image entry
            img_name = f"{sample_id}.jpg"
            img_info = tarfile.TarInfo(name=img_name)
            img_info.size = len(img_bytes)
            img_buffer = BytesIO(img_bytes)
            
            # Add WebDataset required attributes via pax_headers
            img_info.pax_headers = {
                "fname": img_name,
                "__key__": sample_id,
                "__url__": f"file://{img_name}"  # Add URL for consistency
            }
            tar.addfile(img_info, img_buffer)
            
            # Save caption with WebDataset format
            caption = image_captions[image_id][0]
            txt_bytes = caption.encode('utf-8')
            txt_name = f"{sample_id}.txt"
            txt_info = tarfile.TarInfo(name=txt_name)
            txt_info.size = len(txt_bytes)
            txt_buffer = BytesIO(txt_bytes)
            
            # Add WebDataset required attributes via pax_headers
            txt_info.pax_headers = {
                "fname": txt_name,
                "__key__": sample_id,
                "__url__": f"file://{txt_name}"  # Add URL for consistency
            }
            tar.addfile(txt_info, txt_buffer)
    
    # Create MMC4-style dataset
    print("\nCreating MMC4-style dataset...")
    with tarfile.open(os.path.join(output_dir, "mmc4-shard-0000.tar"), "w") as tar:
        successful_samples = 0
        
        for i in range(0, len(mmc4_images), 2):
            if successful_samples >= num_samples_mmc4:
                break
                
            if i + 1 >= len(mmc4_images):
                continue
            
            sample_id = f"{successful_samples:08d}"
            print(f"\nProcessing MMC4 sample {sample_id}")
            
            # Get two images and their captions
            images = []
            captions = []
            
            for j in range(2):
                img = download_coco_image(mmc4_images[i + j])
                if img is not None:
                    images.append(img)
                    captions.append(image_captions[mmc4_images[i + j]][0])
                    print(f"  Added image {j+1} with size {img.size}")
                    print(f"  Caption {j+1}: {captions[-1][:50]}...")
            
            if len(images) < 2:
                print("  Skipping: Not enough valid images")
                continue
            
            # Create MMC4 sample
            data = create_mmc4_sample(images, captions)
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')
            
            # Add to tar with WebDataset format
            json_name = f"{sample_id}.json"
            json_info = tarfile.TarInfo(name=json_name)
            json_info.size = len(json_bytes)
            json_buffer = BytesIO(json_bytes)
            
            # Add WebDataset required attributes via pax_headers
            json_info.pax_headers = {
                "fname": json_name,
                "__key__": sample_id
            }
            tar.addfile(json_info, json_buffer)
            
            successful_samples += 1
            print(f"  Successfully added sample {sample_id}")



if __name__ == "__main__":
    output_dir = "./minimal_datasets"
    create_minimal_coco_dataset(output_dir, num_samples_laion=50, num_samples_mmc4=50)
    print("\nDataset creation completed!")