---
license: mit
---

# Supervised ViT-H/14 (DeiT III) with an ImageNet-1k linear head

A ViT-H/14 trained supervised on ImageNet-1k with DeiT III, with a newly trained linear classification head for the 1000 ImageNet-1k classes, packaged for Brain-Score. It is the supervised control of the I-JEPA ViT-H/14 submission of neuro-jepas.

## Model Details

### Model Description

The ViT-H/14 backbone is `timm/deit3_huge_patch14_224.fb_in1k`, trained supervised on ImageNet-1k with DeiT III (see [timm/deit3_huge_patch14_224.fb_in1k](https://huggingface.co/timm/deit3_huge_patch14_224.fb_in1k)). DeiT III is the only ViT-H/14 trained supervised on ImageNet-1k alone, the training data of the I-JEPA ViT-H/14. Its architecture adds a class token and LayerScale to the I-JEPA ViT-H/14, but the probe reads the same features: the average of the 256 final patch tokens (1280 dimensions), without the class token. The backbone was kept frozen and its classification head was replaced by a linear head trained on ImageNet-1k (linear probe), with the same recipe as the heads of the other ViT-H/14 submissions. The head provides the 1000 ImageNet logits that the behavioral benchmarks of Brain-Score require. For Brain-Score, the patch tokens of every transformer block are also exposed as a 1280 × 16 × 16 grid, so that token activations can be treated like a convolutional feature map.

- **Developed by:** backbone by Meta AI (Touvron et al., 2022); linear head and packaging by Niels Thiemann
- **Shared by:** [niels-c](https://huggingface.co/niels-c)
- **Model type:** Image classifier: supervised Vision Transformer (ViT-H/14) backbone with a linear head (1000 ImageNet-1k classes)
- **License:** MIT; the backbone weights are also subject to the license of their source repository
- **Finetuned from model:** [timm/deit3_huge_patch14_224.fb_in1k](https://huggingface.co/timm/deit3_huge_patch14_224.fb_in1k) (frozen backbone, linear probe)

### Model Sources

- **Repository:** [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions) (Brain-Score plugin), [neuro-jepas](https://github.com/niels-c/neuro-jepas) (training code)
- **Paper:** https://arxiv.org/abs/2204.07118

## Uses

### Direct Use

Classifying images into the 1000 ImageNet-1k classes, and extracting the activations of its layers as a supervised reference for comparison with neural and behavioral data of the primate visual system.

### Downstream Use

The model was built as a submission to [Brain-Score](https://www.brain-score.org/vision/), where it is scored as the plugin `vith14_supervised_linhead` on neural (V1, V2, V4, IT) and behavioral benchmarks. It is part of the neuro-jepas project, which compares joint-embedding predictive architectures (JEPAs) and other self-supervised models on their similarity to the brain.

### Out-of-Scope Use

- Production or safety-critical image classification. The head is a linear probe on a frozen backbone and was not tuned for robustness or calibration.
- Images far from the ImageNet-1k distribution (e.g., medical, satellite or synthetic images) and labels outside the 1000 ImageNet-1k classes.
- Any decision about people. ImageNet-1k classes are not designed to describe people.

## Bias, Risks, and Limitations

- The model inherits the biases of ImageNet-1k, e.g., its class selection, the geographic and demographic skew of its photos and its label noise.
- The architecture is not identical to the I-JEPA ViT-H/14 it is compared with: it has a class token and LayerScale. The class token takes part in attention but is left out of the head and of the Brain-Score layers.
- The original head of the backbone, which reads the class token, was discarded and replaced by the linear probe on the averaged patch tokens, so the accuracy differs from the one published for DeiT III.
- The model expects 224 × 224 RGB images. The probe was evaluated with a resize to 256 pixels and a 224 center crop, whereas Brain-Score resizes the whole image to 224 × 224, which changes the aspect ratio.
- With 632 M parameters and a 2.5 GB checkpoint, inference on CPU (as on Brain-Score) is slow and needs several GB of memory.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Use the preprocessing of the code below. When comparing models on Brain-Score, also compare with the untrained baseline of the same size ([vith14_untrained_linhead](https://huggingface.co/niels-c/vith14_untrained_linhead)), which has the I-JEPA architecture.

## How to Get Started with the Model

Requirements: `pip install torch torchvision "timm>=1.0" huggingface_hub safetensors`

```python
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision.transforms import v2


class ViTH(VisionTransformer):
    """
    DeiT III ViT-H/14 (class token and LayerScale) with a linear head (fc) on the average of the final patch tokens,
    returning logits.
    """

    def __init__(self):
        super().__init__(
            img_size=224,
            patch_size=14,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            no_embed_class=True,
            init_values=1e-6,
            num_classes=0,
            global_pool="avg",
            fc_norm=False,
        )
        self.fc = nn.Linear(self.embed_dim, 1000)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(super().forward(images))


model = ViTH()
model.load_state_dict(
    load_file(
        hf_hub_download(
            "niels-c/vith14_supervised_linhead", "vith14_supervised_linhead.safetensors"
        )
    )
)
model.eval()

# Evaluation transform of the linear probe
preprocess = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

image = preprocess(Image.open("image.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    logits = model(image)  # (1, 1000) ImageNet-1k logits
print(logits.argmax(dim=1))
```

To score the model on Brain-Score, use the plugin `vith14_supervised_linhead` from [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions/tree/main/models/vith14_supervised_linhead), which loads this checkpoint:

```python
import brainscore_vision

model = brainscore_vision.load_model("vith14_supervised_linhead")
```

## Training Details

### Training Data

Linear head: the training split of ImageNet-1k (ILSVRC 2012; 1,281,167 images, 1000 classes) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Backbone: supervised with DeiT III on ImageNet-1k (see [timm/deit3_huge_patch14_224.fb_in1k](https://huggingface.co/timm/deit3_huge_patch14_224.fb_in1k)).

### Training Procedure

The backbone was frozen and only the linear head was trained with the cross-entropy loss, using the trainer of [neuro-jepas](https://github.com/niels-c/neuro-jepas) (experiment `vith14_supervised_linhead`).

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

- **Parameters:** 632,126,440 (encoder 630,845,440, head 1,281,000)
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
| Top-1 accuracy | 83.54% |
| Top-5 accuracy | 96.31% |
| Cross-entropy | 0.650 |

#### Summary

The frozen supervised features reach 83.5% top-1 accuracy with a linear head on ImageNet-1k, the upper reference for the I-JEPA ViT-H/14 (76.9%).

## Technical Specifications

### Model Architecture and Objective

ViT-H/14 encoder of DeiT III as in `timm.create_model("deit3_huge_patch14_224", global_pool="avg", fc_norm=False)` (32 transformer blocks with LayerScale, hidden size 1280, MLP size 5120, 16 attention heads, patch size 14, 224 × 224 inputs, i.e., 256 patch tokens plus a class token), followed by a linear layer `fc` on the average of the final patch tokens.

Backbone: supervised classification on ImageNet-1k labels (DeiT III recipe). Head: cross-entropy on ImageNet-1k labels.

Brain-Score commitments of the plugin `vith14_supervised_linhead`: the recorded layers are the outputs of the 32 transformer blocks without the class token, reshaped to 1280 × 16 × 16 grids (`backbone.blocks.{0..31}.grid`), and the logits (`backbone.fc`), and `backbone.fc` is the behavioral readout. The regions are committed to these layers:

| Region | Layer |
|---|---|
| V1 | `backbone.blocks.5.grid` |
| V2 | `backbone.blocks.13.grid` |
| V4 | `backbone.blocks.21.grid` |
| IT | `backbone.blocks.31.grid` |

### Compute Infrastructure

#### Software

PyTorch, `timm` >= 1.0, safetensors, huggingface_hub; the head was trained with the neuro-jepas trainer.

## Citation

Please cite the paper of the backbone:

**BibTeX:**

```bibtex
@misc{touvron2022deitiiirevengevit,
  title={DeiT III: Revenge of the ViT},
  author={Hugo Touvron and Matthieu Cord and Hervé Jégou},
  year={2022},
  eprint={2204.07118},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2204.07118},
}
```

**APA:**

Touvron, H., Cord, M., & Jégou, H. (2022). *DeiT III: Revenge of the ViT*. arXiv. https://arxiv.org/abs/2204.07118

## Glossary

- **Linear probe:** a linear classifier trained on the features of a frozen network; it measures how linearly separable the classes are in these features.
- **LayerScale:** a learned per-channel scaling of the output of each attention and MLP branch of a transformer block.
- **Brain-Score:** a platform that scores models by the similarity of their activations and behavior to those of primates on visual tasks.
- **Region commitment:** the layer whose activations Brain-Score compares to the recordings from a brain region.

## Model Card Authors

Niels Thiemann

## Model Card Contact

[niels-c](https://huggingface.co/niels-c) on Hugging Face
