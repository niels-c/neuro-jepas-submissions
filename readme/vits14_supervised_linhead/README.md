---
license: mit
---

# Supervised ViT-S/14 with an ImageNet-1k linear head

A ViT-S/14 trained supervised from scratch on ImageNet-1k, with a newly trained linear classification head for the 1000 ImageNet-1k classes, packaged for Brain-Score. It is the supervised control of the I-JEPA ViT-S/14 trained in neuro-jepas, with the same architecture.

## Model Details

### Model Description

The ViT-S/14 backbone (no class token) was trained supervised on ImageNet-1k for 300 epochs in the neuro-jepas project (experiment `vits14_supervised_in1k`), with the DeiT-S recipe but without repeated augmentation. Its architecture is that of the I-JEPA ViT-S/14 of the project. The backbone was then frozen and its classification head was replaced by a linear head on the average of the 256 final patch tokens (384 dimensions), trained on ImageNet-1k (linear probe), so that all ViT-S/14 heads come from the same probe. The head provides the 1000 ImageNet logits that the behavioral benchmarks of Brain-Score require. For Brain-Score, the output of every transformer block is also exposed as a 384 × 16 × 16 grid, so that token activations can be treated like a convolutional feature map.

- **Developed by:** Niels Thiemann (backbone training, linear head, packaging); training recipe by Touvron et al. (2021)
- **Shared by:** [niels-c](https://huggingface.co/niels-c)
- **Model type:** Image classifier: supervised Vision Transformer (ViT-S/14) backbone with a linear head (1000 ImageNet-1k classes)
- **License:** MIT

### Model Sources

- **Repository:** [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions) (Brain-Score plugin), [neuro-jepas](https://github.com/niels-c/neuro-jepas) (training code)
- **Paper:** https://arxiv.org/abs/2012.12877 (DeiT training recipe)

## Uses

### Direct Use

Classifying images into the 1000 ImageNet-1k classes, and extracting the activations of its layers as a supervised reference for comparison with neural and behavioral data of the primate visual system.

### Downstream Use

The model was built as a submission to [Brain-Score](https://www.brain-score.org/vision/), where it is scored as the plugin `vits14_supervised_linhead` on neural (V1, V2, V4, IT) and behavioral benchmarks. It is part of the neuro-jepas project, which compares joint-embedding predictive architectures (JEPAs) and other self-supervised models on their similarity to the brain.

### Out-of-Scope Use

- Production or safety-critical image classification. The head is a linear probe on a frozen backbone and was not tuned for robustness or calibration.
- Images far from the ImageNet-1k distribution (e.g., medical, satellite or synthetic images) and labels outside the 1000 ImageNet-1k classes.
- Any decision about people. ImageNet-1k classes are not designed to describe people.

## Bias, Risks, and Limitations

- The model inherits the biases of ImageNet-1k, e.g., its class selection, the geographic and demographic skew of its photos and its label noise.
- The backbone is one training run (seed 0). It differs from the published DeiT-S in its patch size (14 instead of 16), the missing class token and the missing repeated augmentation.
- The head of the supervised training was discarded and replaced by the linear probe.
- The model expects 224 × 224 RGB images. The probe was evaluated with a resize to 256 pixels and a 224 center crop, whereas Brain-Score resizes the whole image to 224 × 224, which changes the aspect ratio.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Use the preprocessing of the code below.

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


class ViTS(VisionTransformer):
    """
    ViT-S/14 without class token, with a linear head (fc) on the average of the final patch tokens, returning logits.
    """

    def __init__(self):
        super().__init__(
            img_size=224,
            patch_size=14,
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=0,
            class_token=False,
            global_pool="avg",
            fc_norm=False,
        )
        self.fc = nn.Linear(self.embed_dim, 1000)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(super().forward(images))


model = ViTS()
model.load_state_dict(
    load_file(
        hf_hub_download(
            "niels-c/vits14_supervised_linhead", "vits14_supervised_linhead.safetensors"
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

To score the model on Brain-Score, use the plugin `vits14_supervised_linhead` from [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions/tree/main/models/vits14_supervised_linhead), which loads this checkpoint:

```python
import brainscore_vision

model = brainscore_vision.load_model("vits14_supervised_linhead")
```

## Training Details

### Training Data

Backbone and linear head: the training split of ImageNet-1k (ILSVRC 2012; 1,281,167 images, 1000 classes) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

### Training Procedure

Both stages used the trainer of [neuro-jepas](https://github.com/niels-c/neuro-jepas).

**Backbone (experiment `vits14_supervised_in1k`):** supervised training from scratch with the DeiT-S recipe, without repeated augmentation.

- **Epochs:** 300
- **Batch size:** 1024
- **Optimizer:** AdamW, weight decay 0.05
- **Learning rate:** 1e-3 (5e-4 × 1024 / 512) peak after a linear warmup of 5 epochs, cosine decay to 1e-5
- **Loss:** cross-entropy with label smoothing 0.1
- **Augmentation:** random resized crop (scale 0.08 to 1.0), RandAugment (2 operations, magnitude 9), random erasing (probability 0.25), Mixup (alpha 0.8), CutMix (alpha 1.0)
- **Regularization:** stochastic depth 0.1

**Linear head (experiment `vits14_supervised_linhead`):** the backbone was frozen and only the linear head was trained with the cross-entropy loss.

#### Preprocessing

Linear head:

- Random resized crop to 224 × 224 (scale 0.3 to 1.0) and random horizontal flip
- Normalization: ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)

#### Training Hyperparameters

Linear head:

- **Training regime:** bf16 mixed precision
- **Epochs:** 10 (12,510 steps)
- **Batch size:** 1024
- **Optimizer:** AdamW (betas 0.9, 0.999), weight decay 0
- **Learning rate:** 1e-3 peak after a linear warmup of 1 epoch, cosine decay to 1e-6
- **Loss:** cross-entropy without label smoothing
- **Seed:** 0

#### Speeds, Sizes, Times

- **Parameters:** 22,003,816 (encoder 21,618,816, head 385,000)
- **Checkpoint:** 88 MB, float32 safetensors

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The validation split of ImageNet-1k (50,000 images) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k), resized to 256 pixels on the shorter side and center-cropped to 224 × 224, with normalization: ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).

#### Metrics

Top-1 and top-5 accuracy and the cross-entropy loss, the standard metrics of linear probes on ImageNet-1k.

### Results

| Metric | ImageNet-1k validation |
|---|---|
| Top-1 accuracy | 80.51% |
| Top-5 accuracy | 95.05% |
| Cross-entropy | 0.797 |

#### Summary

The frozen supervised features reach 80.5% top-1 accuracy with a linear head on ImageNet-1k, the upper reference for the self-supervised ViT-S/14s of neuro-jepas.

## Technical Specifications

### Model Architecture and Objective

ViT-S/14 as in `timm.create_model("vit_small_patch16_224", patch_size=14, class_token=False, global_pool="avg", fc_norm=False)` (12 transformer blocks, hidden size 384, MLP size 1536, 6 attention heads, patch size 14, 224 × 224 inputs, i.e., 256 patch tokens without a class token), followed by a linear layer `fc` on the average of the final patch tokens.

Backbone: supervised classification on ImageNet-1k labels. Head: cross-entropy on ImageNet-1k labels.

Brain-Score commitments of the plugin `vits14_supervised_linhead`: the recorded layers are the outputs of the 12 transformer blocks, reshaped to 384 × 16 × 16 grids (`backbone.blocks.{0..11}.grid`), and the logits (`backbone.fc`), and `backbone.fc` is the behavioral readout. The regions are committed to these layers:

| Region | Layer |
|---|---|
| V1 | `backbone.blocks.1.grid` |
| V2 | `backbone.blocks.4.grid` |
| V4 | `backbone.blocks.7.grid` |
| IT | `backbone.blocks.11.grid` |

### Compute Infrastructure

#### Software

PyTorch, `timm` >= 1.0, safetensors, huggingface_hub; backbone and head were trained with the neuro-jepas trainer.

## Citation

Please cite the paper of the training recipe:

**BibTeX:**

```bibtex
@inproceedings{touvron2021training,
  title={Training data-efficient image transformers \& distillation through attention},
  author={Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and J{\'e}gou, Herv{\'e}},
  booktitle={International Conference on Machine Learning},
  pages={10347--10357},
  year={2021},
  organization={PMLR},
}
```

**APA:**

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). Training data-efficient image transformers & distillation through attention. In *International Conference on Machine Learning* (pp. 10347–10357). PMLR.

## Glossary

- **Linear probe:** a linear classifier trained on the features of a frozen network; it measures how linearly separable the classes are in these features.
- **Brain-Score:** a platform that scores models by the similarity of their activations and behavior to those of primates on visual tasks.
- **Region commitment:** the layer whose activations Brain-Score compares to the recordings from a brain region.

## Model Card Authors

Niels Thiemann

## Model Card Contact

[niels-c](https://huggingface.co/niels-c) on Hugging Face
