---
license: mit
---

# ResNet-50 Barlow Twins with an ImageNet-1k linear head

A ResNet-50 pretrained with Barlow Twins, with a linear classification head for the 1000 ImageNet-1k classes, packaged for Brain-Score.

## Model Details

### Model Description

The ResNet-50 backbone was trained self-supervised with Barlow Twins on ImageNet-1k (see [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins)). The backbone weights are those released in [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins) and were kept frozen. A linear head on the globally average-pooled output of the last stage (2048 dimensions) was trained on ImageNet-1k (linear probe). The head provides the 1000 ImageNet logits that the behavioral benchmarks of Brain-Score require.

- **Developed by:** backbone by Meta AI (Zbontar et al., 2021); linear head and packaging by Niels Thiemann
- **Shared by:** [niels-c](https://huggingface.co/niels-c)
- **Model type:** Image classifier: self-supervised convolutional ResNet-50 backbone with a linear head (1000 ImageNet-1k classes)
- **License:** MIT; the backbone weights are also subject to the license of their source repository
- **Finetuned from model:** [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins) (frozen backbone, linear probe)

### Model Sources

- **Repository:** [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions) (Brain-Score plugin), [neuro-jepas](https://github.com/niels-c/neuro-jepas) (training code)
- **Paper:** https://arxiv.org/abs/2103.03230

## Uses

### Direct Use

Classifying images into the 1000 ImageNet-1k classes, and extracting the activations of its layers for comparison with neural and behavioral data of the primate visual system.

### Downstream Use

The model was built as a submission to [Brain-Score](https://www.brain-score.org/vision/), where it is scored as the plugin `resnet50_barlowtwins_linhead` on neural (V1, V2, V4, IT) and behavioral benchmarks. It is part of the neuro-jepas project, which compares joint-embedding predictive architectures (JEPAs) and other self-supervised models on their similarity to the brain.

### Out-of-Scope Use

- Production or safety-critical image classification. The head is a linear probe on a frozen backbone and was not tuned for robustness or calibration.
- Images far from the ImageNet-1k distribution (e.g., medical, satellite or synthetic images) and labels outside the 1000 ImageNet-1k classes.
- Any decision about people. ImageNet-1k classes are not designed to describe people.

## Bias, Risks, and Limitations

- The model inherits the biases of ImageNet-1k, e.g., its class selection, the geographic and demographic skew of its photos and its label noise.
- Only the linear head was trained; the backbone features were not adapted to classification, so accuracy is lower than for a fine-tuned model.
- The model expects 224 × 224 RGB images. The probe was evaluated with a resize to 256 pixels and a 224 center crop, whereas Brain-Score resizes the whole image to 224 × 224, which changes the aspect ratio.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Use the preprocessing of the code below. When comparing models on Brain-Score, also compare with the untrained baseline of the same architecture ([resnet50_untrained_linhead](https://huggingface.co/niels-c/resnet50_untrained_linhead)).

## How to Get Started with the Model

Requirements: `pip install torch torchvision huggingface_hub safetensors`

```python
import torch
import torchvision
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import v2

model = torchvision.models.resnet50(weights=None)
model.load_state_dict(load_file(hf_hub_download("niels-c/resnet50_barlowtwins_linhead", "resnet50_barlowtwins_linhead.safetensors")))
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

To score the model on Brain-Score, use the plugin `resnet50_barlowtwins_linhead` from [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions/tree/main/models/resnet50_barlowtwins_linhead), which pins the revision of this checkpoint:

```python
import brainscore_vision

model = brainscore_vision.load_model("resnet50_barlowtwins_linhead")
```

## Training Details

### Training Data

Linear head: the training split of ImageNet-1k (ILSVRC 2012; 1,281,167 images, 1000 classes) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Backbone: self-supervised with Barlow Twins on ImageNet-1k (see [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins)).

### Training Procedure

The backbone was frozen (batch normalization statistics included) and only the linear head was trained with the cross-entropy loss, using the trainer of [neuro-jepas](https://github.com/niels-c/neuro-jepas) (experiment `resnet50_barlowtwins_linhead`).

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

- **Parameters:** 25,557,032 (backbone 23,508,032, head 2,049,000)
- **Checkpoint:** 103 MB, float32 safetensors

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The validation split of ImageNet-1k (50,000 images) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k), resized to 256 pixels on the shorter side and center-cropped to 224 × 224, with normalization: ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).

#### Metrics

Top-1 and top-5 accuracy and the cross-entropy loss, the standard metrics of linear probes on ImageNet-1k.

### Results

| Metric | ImageNet-1k validation |
|---|---|
| Top-1 accuracy | 71.60% |
| Top-5 accuracy | 90.02% |
| Cross-entropy | 1.173 |

#### Summary

The frozen Barlow Twins features reach 71.6% top-1 accuracy with a linear head on ImageNet-1k.

## Technical Specifications

### Model Architecture and Objective

ResNet-50 as in `torchvision.models.resnet50` (bottleneck blocks in 4 stages of 3, 4, 6 and 3 blocks), whose layer `fc` is the linear head on the globally average-pooled 2048-dimensional features.

Backbone: self-supervised Barlow Twins objective. Head: cross-entropy on ImageNet-1k labels.

Brain-Score commitments of the plugin `resnet50_barlowtwins_linhead`: the recorded layers are `backbone.conv1`, `backbone.layer1`, `backbone.layer2`, `backbone.layer3`, `backbone.layer4`, `backbone.fc`, and `backbone.fc` is the behavioral readout. The regions are committed to these layers:

| Region | Layer |
|---|---|
| V1 | `backbone.layer1` |
| V2 | `backbone.layer2` |
| V4 | `backbone.layer3` |
| IT | `backbone.layer4` |

### Compute Infrastructure

#### Software

PyTorch, torchvision (inference), `timm` (training), safetensors, huggingface_hub; the head was trained with the neuro-jepas trainer.

## Citation

Please cite the paper of the backbone:

**BibTeX:**

```bibtex
@misc{zbontar2021barlowtwinsselfsupervisedlearning,
  title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  author={Jure Zbontar and Li Jing and Ishan Misra and Yann LeCun and Stéphane Deny},
  year={2021},
  eprint={2103.03230},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2103.03230},
}
```

**APA:**

Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). *Barlow Twins: Self-supervised learning via redundancy reduction*. arXiv. https://arxiv.org/abs/2103.03230

## Glossary

- **Linear probe:** a linear classifier trained on the features of a frozen network; it measures how linearly separable the classes are in these features.
- **Brain-Score:** a platform that scores models by the similarity of their activations and behavior to those of primates on visual tasks.
- **Region commitment:** the layer whose activations Brain-Score compares to the recordings from a brain region.

## Model Card Authors

Niels Thiemann

## Model Card Contact

[niels-c](https://huggingface.co/niels-c) on Hugging Face
