# Simple Flamingo Implementation

<p align="center">
  <img src="assets/flamingo_logo.png" width="200" alt="Flamingo Logo"/>
</p>

This is a simplified reimplementation of the Flamingo model, focused on single GPU training with smaller datasets for educational purposes. The goal is to understand the core mechanics without the complexity of distributed training.

Inspired by:
- [Open Flamingo](https://github.com/mlfoundations/open_flamingo)
- [Mini Flamingo](https://github.com/dhansmair/flamingo-mini/)

## Model Architecture

### Vision Components
- **Vision Encoder**: CLIP ViT-L-14
- **Input Shape**: `(B=2, T_img=1, F=1, C=3, H=224, W=224)`
  - B: batch size
  - T_img: temporal dimension
  - F: frames
  - C: channels
  - H/W: height/width

### Language Components
- **Language Model**: OPT-125m
- **Tokenizer**: OPT-125m tokenizer with special tokens:
  - `<|endofchunk|>`: Marks end of image-text pairs
  - `<image>`: Marks image positions in text

## Model Workflow

### 1. Vision Processing
```python
# Vision Encoding
input_image = (2, 1, 1, 3, 224, 224)
vision_features = vision_encoder(input_image)  # CLIP ViT-L-14
# Output shape: (2, 256, 1024)  # (batch, sequence_length, hidden_dim)

# Reshape for perceiver
vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
# Shape: (2, 1, 1, 256, 1024)
```

### 2. Perceiver Processing
```python
# PerceiverResampler converts variable-length visual features to fixed length
perceiver_output = perceiver(vision_features)
# Output shape: (2, 1, 64, 1024)  # (batch, temporal, num_latents, dim)
```

### 3. Language Model Architecture

The OPT-125m language model is modified with:
1. FlamingoLMMixin extension
2. GatedCrossAttention layers before every transformer decoder layer
3. Flamingo layer wrapping of original decoder layers

#### Detailed Processing Flow
```python
# Input Processing
inputs_ids=[2x20]  # Initial input shape
inputs_embeds = self.embed_tokens(input_ids)  # Shape: [2x20x768]
position_embedding = OPTLearnedPositionalEmbedding(2050, 768)

# For each decoder layer:
1. Gated cross attention
2. Normal decoder layer processing

lang_x = self.gated_cross_attn_layer(
    lang_x,
    self.vis_x,
    media_locations=self.media_locations,
    use_cached_media=self.use_cached_media,
)
lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask)
```

### Special(Vision) Token Handling

The model processes special tokens by:
1. Detecting `<image>` token positions
2. Applying cross-attention from text to images only at these positions

#### Visual Features Integration
```python
# Add visual features to each decoder layer
for layer in self.lang_encoder._get_decoder_layers():
    layer.condition_vis_x(vision_x)

# Set media locations for cross attention
for layer in self.lang_encoder._get_decoder_layers():
    layer.condition_media_locations(media_locations)
```

### 4. Cross-Attention Mechanism

The gated cross-attention allows the model to selectively attend to visual features:

```python
def forward(self, x, media, media_locations):
    # Input shapes:
    # x: [B, T_txt, D_txt] = [1, 8, 1024]        # 8 text tokens
    # media: [B, T_img, n, D] = [1, 3, 64, 1024]  # 3 images, 64 tokens each
    # media_locations: [B, T_txt] = [1, 8]        # Binary mask for <image> positions

    # 1. Create Q, K, V
    q = self.to_q(x)                          # [1, 8, 512] (if heads=8, dim_head=64)
    media = rearrange(media, "b t n d -> b (t n) d")  # [1, 192, 1024]
    k, v = self.to_kv(media).chunk(2, dim=-1)  # Each: [1, 192, 512]

    # 2. Reshape for multi-head attention (h=8 heads)
    q = rearrange(q, "b n (h d) -> b h n d", h=h)     # [1, 8, 8, 64]
    k = rearrange(k, "b n (h d) -> b h n d", h=h)     # [1, 8, 192, 64]
    v = rearrange(v, "b n (h d) -> b h n d", h=h)     # [1, 8, 192, 64]

    # 3. Calculate attention scores and apply mask
    sim = einsum("... i d, ... j d -> ... i j", q, k)  # [1, 8, 8, 192]
    sim = sim.masked_fill(~text_to_media_mask, -inf)   # [1, 8, 8, 192]
    attn = sim.softmax(dim=-1)                         # [1, 8, 8, 192]

    # 4. Final output
    out = einsum("... i j, ... j d -> ... i d", attn, v)  # [1, 8, 8, 64]
    out = rearrange(out, "b h n d -> b n (h d)")          # [1, 8, 512]
```


- Text tokens only attend to valid media locations
- Example with media_locations = [1, 0, 0, 1, 0, 0]:
- text_time becomes [1, 1, 1, 2, 2, 2]
- media_time = [1, 2]
- First three tokens attend to first image
- Last three tokens attend to second image



## Model Creation


```python
# 1. Initialize base language model
lang_encoder = AutoModelForCausalLM.from_pretrained('opt-125m')

# 2. Add Flamingo capabilities
extend_instance(lang_encoder, FlamingoLMMixin)

# 3. Create Flamingo model
model = Flamingo(
    vision_encoder=vision_encoder,
    lang_encoder=lang_encoder,
    ...
)
```

```python
# Original LM structure:
TransformerLayer1 -> TransformerLayer2 -> TransformerLayer3 -> ...

# After Flamingo modification (decoder layers only):
FlamingoLayer(
    GatedCrossAttention
    TransformerLayer1,
) -> 
FlamingoLayer(
    GatedCrossAttention,
    TransformerLayer2, 
) -> 
FlamingoLayer(
    GatedCrossAttention,
    TransformerLayer3,
) -> ...
```


## Loss Calculation

```python
if labels is not None:
    # Prepare labels
    labels = labels.to(logits.device)
    
    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate cross entropy loss
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, self.config.vocab_size),
        shift_labels.view(-1)
    )
```
