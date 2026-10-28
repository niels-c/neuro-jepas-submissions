---
license: mit
---

# Untrained ViT-H/14 with an ImageNet-1k linear head

A randomly initialized, untrained ViT-H/14 with a linear classification head trained on ImageNet-1k. It is the untrained baseline of the neuro-jepas Brain-Score submissions.

## Model Details

### Model Description

The ViT-H/14 backbone keeps its random initialization (the default initialization of `transformers` for the architecture of facebook/ijepa_vith14_1k, seed 0) and was never trained. Only a linear head on the average of the 256 final patch tokens (1280 dimensions) was trained on ImageNet-1k with the backbone frozen. The model measures how much of a Brain-Score result is explained by the architecture alone, and serves as a baseline for the self-supervised models of the same architecture. For Brain-Score, the output of every transformer block is also exposed as a 1280 × 16 × 16 grid, so that token activations can be treated like a convolutional feature map.

- **Developed by:** Niels Thiemann (linear head, packaging)
- **Shared by:** [niels-c](https://huggingface.co/niels-c)
- **Model type:** Image classifier: untrained Vision Transformer (ViT-H/14) backbone with a linear head (1000 ImageNet-1k classes)
- **License:** MIT

### Model Sources

- **Repository:** [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions) (Brain-Score plugin), [neuro-jepas](https://github.com/niels-c/neuro-jepas) (training code)

## Uses

### Direct Use

Extracting the activations of its layers as an untrained baseline for comparison with neural and behavioral data of the primate visual system.

### Downstream Use

The model was built as a submission to [Brain-Score](https://www.brain-score.org/vision/), where it is scored as the plugin `vith14_untrained_linhead` on neural (V1, V2, V4, IT) and behavioral benchmarks. It is part of the neuro-jepas project, which compares joint-embedding predictive architectures (JEPAs) and other self-supervised models on their similarity to the brain.

### Out-of-Scope Use

- Classification of any kind: with 5.5% top-1 accuracy on ImageNet-1k, the model is a baseline, not a usable classifier.
- Production or safety-critical image classification. The head is a linear probe on a frozen backbone and was not tuned for robustness or calibration.
- Images far from the ImageNet-1k distribution (e.g., medical, satellite or synthetic images) and labels outside the 1000 ImageNet-1k classes.
- Any decision about people. ImageNet-1k classes are not designed to describe people.

## Bias, Risks, and Limitations

- The model inherits the biases of ImageNet-1k, e.g., its class selection, the geographic and demographic skew of its photos and its label noise.
- Only the linear head was trained; the backbone features were not adapted to classification, so accuracy is lower than for a fine-tuned model.
- The model expects 224 × 224 RGB images. The probe was evaluated with a resize to 256 pixels and a 224 center crop, whereas Brain-Score resizes the whole image to 224 × 224, which changes the aspect ratio.
- With 632 M parameters and a 2.5 GB checkpoint, inference on CPU (as on Brain-Score) is slow and needs several GB of memory.
- The weights are one random draw (seed 0); other initializations would give somewhat different results.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Use the preprocessing of the code below. Compare results with the pretrained models of the same architecture rather than reading them on their own.

## How to Get Started with the Model

Requirements: `pip install torch torchvision "transformers>=5.19.0" huggingface_hub safetensors`

```python
import torch
from torch import nn
from transformers import IJepaConfig, IJepaModel
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import v2

NUM_LAYERS, NUM_CLASSES = 32, 1000
# Architecture of facebook/ijepa_vith14_1k
CONFIG = IJepaConfig(
    hidden_size=1280,
    intermediate_size=5120,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=16,
    patch_size=14,
    image_size=224,
    layer_norm_eps=1e-6,
)


class IJepaClassifier(IJepaModel):
    """
    I-JEPA encoder with a linear head (fc) on the average of the final patch tokens, returning logits.
    """

    def __init__(self, config: IJepaConfig):
        super().__init__(config)
        self.fc = nn.Linear(config.hidden_size, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(super().forward(pixel_values=images).last_hidden_state.mean(dim=1))


model = IJepaClassifier(CONFIG)
model.load_state_dict(load_file(hf_hub_download("niels-c/vith14_untrained_linhead", "vith14_untrained_linhead.safetensors")))
model.eval()

# Evaluation transform of the linear probe
preprocess = v2.Compose([
    v2.ToImage(),
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

image = preprocess(Image.open("image.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    logits = model(image)  # (1, 1000) ImageNet-1k logits
print(logits.argmax(dim=1))
```

To score the model on Brain-Score, use the plugin `vith14_untrained_linhead` from [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions/tree/main/models/vith14_untrained_linhead), which pins the revision of this checkpoint:

```python
import brainscore_vision

model = brainscore_vision.load_model("vith14_untrained_linhead")
```

## Training Details

### Training Data

Linear head: the training split of ImageNet-1k (ILSVRC 2012; 1,281,167 images, 1000 classes) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

### Training Procedure

The backbone was frozen (batch normalization statistics included) and only the linear head was trained with the cross-entropy loss, using the trainer of [neuro-jepas](https://github.com/niels-c/neuro-jepas) (experiment `vith14_untrained_linhead`).

#### Preprocessing

- Random resized crop to 224 × 224 (scale 0.3 to 1.0) and random horizontal flip
- Normalization: ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision
- **Epochs:** 10 (12,510 steps)
- **Batch size:** 1024
- **Optimizer:** AdamW (betas 0.9, 0.999), weight decay 0
- **Learning rate:** 1e-3 peak after a linear warmup of 1 epoch, cosine decay to 1e-6
- **Loss:** cross-entropy without label smoothing
- **Seed:** 0

#### Speeds, Sizes, Times

- **Parameters:** 632,043,240 (encoder 630,762,240, head 1,281,000)
- **Checkpoint:** 2.53 GB, float32 safetensors

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The validation split of ImageNet-1k (50,000 images) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k), resized to 256 pixels on the shorter side and center-cropped to 224 × 224, with normalization: ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).

#### Metrics

Top-1 and top-5 accuracy and the cross-entropy loss, the standard metrics of linear probes on ImageNet-1k.

### Results

| Metric | ImageNet-1k validation |
|---|---|
| Top-1 accuracy | 5.49% |
| Top-5 accuracy | 14.15% |
| Cross-entropy | 5.879 |

#### Summary

The untrained backbone gives features that a linear head can only weakly classify, as expected for a random-weights baseline.

## Technical Specifications

### Model Architecture and Objective

ViT-H/14 encoder as in `transformers.IJepaModel` (32 transformer blocks, hidden size 1280, MLP size 5120, 16 attention heads, patch size 14, 224 × 224 inputs, i.e., 256 patch tokens without a class token), followed by a linear layer `fc` on the average of the final patch tokens.

No pretraining objective (random weights). The head was trained with cross-entropy on ImageNet-1k labels.

Brain-Score commitments of the plugin `vith14_untrained_linhead`: the recorded layers are the outputs of the 32 transformer blocks, reshaped to 1280 × 16 × 16 grids (`backbone.layers.{0..31}.grid`), and the logits (`backbone.fc`), and `backbone.fc` is the behavioral readout. The regions are committed to these layers:

| Region | Layer |
|---|---|
| V1 | `backbone.layers.5.grid` |
| V2 | `backbone.layers.13.grid` |
| V4 | `backbone.layers.21.grid` |
| IT | `backbone.layers.31.grid` |

### Compute Infrastructure

#### Software

PyTorch, `transformers` >= 5.19 (I-JEPA encoder), safetensors, huggingface_hub; the head was trained with the neuro-jepas trainer.

## Glossary

- **Linear probe:** a linear classifier trained on the features of a frozen network; it measures how linearly separable the classes are in these features.
- **Brain-Score:** a platform that scores models by the similarity of their activations and behavior to those of primates on visual tasks.
- **Region commitment:** the layer whose activations Brain-Score compares to the recordings from a brain region.

## Model Card Authors

Niels Thiemann

## Model Card Contact

[niels-c](https://huggingface.co/niels-c) on Hugging Face
