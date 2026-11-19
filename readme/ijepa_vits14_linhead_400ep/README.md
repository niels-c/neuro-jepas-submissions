---
license: mit
---

# I-JEPA ViT-S/14 (400 epochs) with an ImageNet-1k linear head

A ViT-S/14 pretrained self-supervised with I-JEPA (Joint-Embedding Predictive Architecture) on ImageNet-1k for 400 epochs with a recipe retuned for the ViT-S, with a linear classification head for the 1000 ImageNet-1k classes, packaged for Brain-Score. No public I-JEPA ViT-S/14 exists, so the backbone was pretrained in the neuro-jepas project.

## Model Details

### Model Description

The ViT-S/14 backbone (no class token) was pretrained with I-JEPA on ImageNet-1k for 400 epochs. The recipe starts from the official ImageNet-1k ViT-H/14 configuration (`in1k_vith14_ep300.yaml`) and was retuned for the smaller encoder: half the batch size, a twice higher learning rate, a smaller context block, a shallower predictor and a slower EMA. As in I-JEPA, the exported backbone is the EMA target encoder. The backbone was then frozen and a linear head on the average of the 256 final patch tokens (384 dimensions) was trained on ImageNet-1k (linear probe). The head provides the 1000 ImageNet logits that the behavioral benchmarks of Brain-Score require. For Brain-Score, the output of every transformer block is also exposed as a 384 × 16 × 16 grid, so that token activations can be treated like a convolutional feature map.

- **Developed by:** Niels Thiemann (pretraining, linear head, packaging); I-JEPA method and recipe by Assran et al. (2023)
- **Shared by:** [niels-c](https://huggingface.co/niels-c)
- **Model type:** Image classifier: self-supervised Vision Transformer (ViT-S/14) backbone with a linear head (1000 ImageNet-1k classes)
- **License:** MIT

### Model Sources

- **Repository:** [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions) (Brain-Score plugin); the I-JEPA training code was adapted from [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)
- **Paper:** https://arxiv.org/abs/2301.08243

## Uses

### Direct Use

Classifying images into the 1000 ImageNet-1k classes, and extracting the activations of its layers for comparison with neural and behavioral data of the primate visual system.

### Downstream Use

The model was built as a submission to [Brain-Score](https://www.brain-score.org/vision/), where it is scored as the plugin `ijepa_vits14_linhead_400ep` on neural (V1, V2, V4, IT) and behavioral benchmarks. It is part of the neuro-jepas project, which compares joint-embedding predictive architectures (JEPAs) and other self-supervised models on their similarity to the brain.

### Out-of-Scope Use

- Image classification of any kind where accuracy matters. With 37% top-1 accuracy, the linear head is far from usable as a classifier.
- Production or safety-critical use. The head is a linear probe on a frozen backbone and was not tuned for robustness or calibration.
- Images far from the ImageNet-1k distribution (e.g., medical, satellite or synthetic images) and labels outside the 1000 ImageNet-1k classes.
- Any decision about people. ImageNet-1k classes are not designed to describe people.

## Bias, Risks, and Limitations

- The model inherits the biases of ImageNet-1k, e.g., its class selection, the geographic and demographic skew of its photos and its label noise.
- The backbone is one training run (seed 0).
- I-JEPA was published for ViT-H and ViT-g only. The recipe of this ViT-S/14 was retuned by hand, not by a systematic search, and its features are much less linearly separable than those of the published models.
- The implementation deviates from the official one in the learning-rate warmup, which starts from 0 instead of a start learning rate.
- Only the linear head was trained; the backbone features were not adapted to classification.
- The model expects 224 × 224 RGB images. The probe was evaluated with a resize to 256 pixels and a 224 center crop, whereas Brain-Score resizes the whole image to 224 × 224, which changes the aspect ratio.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Use the preprocessing of the code below. When comparing models on Brain-Score, also compare with the supervised model of the same architecture ([vits14_supervised_linhead](https://huggingface.co/niels-c/vits14_supervised_linhead)).

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
            "niels-c/ijepa_vits14_linhead_400ep", "ijepa_vits14_linhead_400ep.safetensors"
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

To score the model on Brain-Score, use the plugin `ijepa_vits14_linhead_400ep` from [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions/tree/main/models/ijepa_vits14_linhead_400ep), which loads this checkpoint:

```python
import brainscore_vision

model = brainscore_vision.load_model("ijepa_vits14_linhead_400ep")
```

## Training Details

### Training Data

Backbone and linear head: the training split of ImageNet-1k (ILSVRC 2012; 1,281,167 images, 1000 classes) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k). The backbone used the images only, without labels.

### Training Procedure

**Backbone:** self-supervised I-JEPA. The encoder sees only the patches of a context block, a narrow ViT predictor predicts the representations of 4 target blocks from it, and the predictions are regressed onto the layer-normalized representations of the full image from the EMA target encoder.

- **Epochs:** 400
- **Batch size:** 1024 (4 gradient-accumulation steps)
- **Optimizer:** AdamW, weight decay 0.04 to 0.4 (cosine)
- **Learning rate:** 2e-3 peak after a linear warmup of 40 epochs, cosine decay to 5e-6
- **EMA momentum of the target encoder:** 0.998 to 1.0
- **Loss:** smooth L1 between predicted and target representations
- **Masking:** 4 target blocks (scale 0.15 to 0.2, aspect ratio 0.75 to 1.5), one context block (scale 0.5 to 0.7) without the target patches, at least 10 patches kept
- **Predictor:** ViT with 6 blocks of width 384
- **Augmentation:** random resized crop (scale 0.3 to 1.0), no flip and no color augmentation

**Linear head:** the backbone was frozen and only the linear head was trained with the cross-entropy loss.

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
| Top-1 accuracy | 37.07% |
| Top-5 accuracy | 61.65% |
| Cross-entropy | 3.019 |

#### Summary

The frozen I-JEPA features reach 37.1% top-1 accuracy with a linear head on ImageNet-1k, up from 27.2% for the checkpoint of the same run after 224 epochs ([ijepa_vits14_linhead](https://huggingface.co/niels-c/ijepa_vits14_linhead)), but far below the 80.5% of the supervised ViT-S/14 with the same architecture and probe ([vits14_supervised_linhead](https://huggingface.co/niels-c/vits14_supervised_linhead)).

## Technical Specifications

### Model Architecture and Objective

ViT-S/14 as in `timm.create_model("vit_small_patch16_224", patch_size=14, class_token=False, global_pool="avg", fc_norm=False)` (12 transformer blocks, hidden size 384, MLP size 1536, 6 attention heads, patch size 14, 224 × 224 inputs, i.e., 256 patch tokens without a class token), followed by a linear layer `fc` on the average of the final patch tokens.

Backbone: I-JEPA, prediction of the representations of masked image blocks in latent space. Head: cross-entropy on ImageNet-1k labels.

Brain-Score commitments of the plugin `ijepa_vits14_linhead_400ep`: the recorded layers are the outputs of the 12 transformer blocks, reshaped to 384 × 16 × 16 grids (`backbone.blocks.{0..11}.grid`), and the logits (`backbone.fc`), and `backbone.fc` is the behavioral readout. The regions are committed to these layers, scaled by depth from the region map of the I-JEPA ViT-H/14:

| Region | Layer |
|---|---|
| V1 | `backbone.blocks.1.grid` |
| V2 | `backbone.blocks.4.grid` |
| V4 | `backbone.blocks.7.grid` |
| IT | `backbone.blocks.11.grid` |

### Compute Infrastructure

#### Software

PyTorch, `timm` >= 1.0, safetensors, huggingface_hub.

## Citation

Please cite the I-JEPA paper:

**BibTeX:**

```bibtex
@misc{assran2023selfsupervisedlearningimagesjointembedding,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Mahmoud Assran and Quentin Duval and Ishan Misra and Piotr Bojanowski and Pascal Vincent and Michael Rabbat and Yann LeCun and Nicolas Ballas},
  year={2023},
  eprint={2301.08243},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2301.08243},
}
```

**APA:**

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. *arXiv preprint arXiv:2301.08243*.

## Glossary

- **I-JEPA:** a self-supervised method that predicts the representations of masked image blocks from a visible context block, in latent space instead of pixel space.
- **EMA target encoder:** a copy of the encoder whose weights are an exponential moving average of the trained encoder; it provides the prediction targets and is the exported backbone.
- **Linear probe:** a linear classifier trained on the features of a frozen network; it measures how linearly separable the classes are in these features.
- **Brain-Score:** a platform that scores models by the similarity of their activations and behavior to those of primates on visual tasks.
- **Region commitment:** the layer whose activations Brain-Score compares to the recordings from a brain region.

## Model Card Authors

Niels Thiemann

## Model Card Contact

[niels-c](https://huggingface.co/niels-c) on Hugging Face
