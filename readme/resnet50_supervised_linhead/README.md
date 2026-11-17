---
license: mit
---

# Supervised ResNet-50 with an ImageNet-1k linear head

A ResNet-50 trained supervised on ImageNet-1k, with a newly trained linear classification head for the 1000 ImageNet-1k classes, packaged for Brain-Score. It is the supervised control of the ResNet-50 submissions of neuro-jepas.

## Model Details

### Model Description

The ResNet-50 backbone is `timm/resnet50.a1_in1k`, trained supervised on ImageNet-1k with procedure A1 of "ResNet strikes back" (see [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k)). Its weights were kept frozen and its classification head was replaced: a new linear head on the globally average-pooled output of the last stage (2048 dimensions) was trained on ImageNet-1k (linear probe), with the same recipe as the heads of the self-supervised ResNet-50s, so that all ResNet-50 submissions differ only in their backbone. The head provides the 1000 ImageNet logits that the behavioral benchmarks of Brain-Score require.

- **Developed by:** backbone by Wightman et al. (2021) in `timm`, architecture by He et al. (2015); linear head and packaging by Niels Thiemann
- **Shared by:** [niels-c](https://huggingface.co/niels-c)
- **Model type:** Image classifier: supervised convolutional ResNet-50 backbone with a linear head (1000 ImageNet-1k classes)
- **License:** MIT; the backbone weights are also subject to the license of their source repository
- **Finetuned from model:** [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) (frozen backbone, linear probe)

### Model Sources

- **Repository:** [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions) (Brain-Score plugin), [neuro-jepas](https://github.com/niels-c/neuro-jepas) (training code)
- **Paper:** https://arxiv.org/abs/1512.03385 (ResNet), https://arxiv.org/abs/2110.00476 (training procedure of the backbone)

## Uses

### Direct Use

Classifying images into the 1000 ImageNet-1k classes, and extracting the activations of its layers as a supervised reference for comparison with neural and behavioral data of the primate visual system.

### Downstream Use

The model was built as a submission to [Brain-Score](https://www.brain-score.org/vision/), where it is scored as the plugin `resnet50_supervised_linhead` on neural (V1, V2, V4, IT) and behavioral benchmarks. It is part of the neuro-jepas project, which compares joint-embedding predictive architectures (JEPAs) and other self-supervised models on their similarity to the brain.

### Out-of-Scope Use

- Production or safety-critical image classification. The head is a linear probe on a frozen backbone and was not tuned for robustness or calibration.
- Images far from the ImageNet-1k distribution (e.g., medical, satellite or synthetic images) and labels outside the 1000 ImageNet-1k classes.
- Any decision about people. ImageNet-1k classes are not designed to describe people.

## Bias, Risks, and Limitations

- The model inherits the biases of ImageNet-1k, e.g., its class selection, the geographic and demographic skew of its photos and its label noise.
- The backbone was trained with labels, unlike the self-supervised ResNet-50s it is compared with, and with a modern, heavily augmented recipe (A1: 600 epochs, LAMB, binary cross-entropy, RandAugment, Mixup and CutMix), not the original ResNet-50 recipe.
- The original head of the backbone was discarded and replaced by the linear probe, so the accuracy differs slightly from the one published for `resnet50.a1_in1k`.
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
model.load_state_dict(
    load_file(
        hf_hub_download(
            "niels-c/resnet50_supervised_linhead",
            "resnet50_supervised_linhead.safetensors",
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

To score the model on Brain-Score, use the plugin `resnet50_supervised_linhead` from [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions/tree/main/models/resnet50_supervised_linhead), which loads this checkpoint:

```python
import brainscore_vision

model = brainscore_vision.load_model("resnet50_supervised_linhead")
```

## Training Details

### Training Data

Linear head: the training split of ImageNet-1k (ILSVRC 2012; 1,281,167 images, 1000 classes) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Backbone: supervised on ImageNet-1k (see [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k)).

### Training Procedure

The backbone was frozen (batch normalization statistics included) and only the linear head was trained with the cross-entropy loss, using the trainer of [neuro-jepas](https://github.com/niels-c/neuro-jepas) (experiment `resnet50_supervised_linhead`).

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
- **Checkpoint:** 102 MB, float32 safetensors

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The validation split of ImageNet-1k (50,000 images) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k), resized to 256 pixels on the shorter side and center-cropped to 224 × 224, with normalization: ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).

#### Metrics

Top-1 and top-5 accuracy and the cross-entropy loss, the standard metrics of linear probes on ImageNet-1k.

### Results

| Metric | ImageNet-1k validation |
|---|---|
| Top-1 accuracy | 79.52% |
| Top-5 accuracy | 94.42% |
| Cross-entropy | 0.840 |

#### Summary

The frozen supervised features reach 79.5% top-1 accuracy with a linear head on ImageNet-1k, the upper reference for the self-supervised ResNet-50s.

## Technical Specifications

### Model Architecture and Objective

ResNet-50 as in `torchvision.models.resnet50` (bottleneck blocks in 4 stages of 3, 4, 6 and 3 blocks), whose layer `fc` is the linear head on the globally average-pooled 2048-dimensional features. The head was trained on `timm.create_model("resnet50")`, whose parameter names are the same.

Backbone: supervised classification on ImageNet-1k labels. Head: cross-entropy on ImageNet-1k labels.

Brain-Score commitments of the plugin `resnet50_supervised_linhead`: the recorded layers are `backbone.conv1`, `backbone.layer1`, `backbone.layer2`, `backbone.layer3`, `backbone.layer4`, `backbone.fc`, and `backbone.fc` is the behavioral readout. The regions are committed to these layers:

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

Please cite the papers of the architecture and of the backbone's training procedure:

**BibTeX:**

```bibtex
@misc{he2015deepresiduallearningimage,
  title={Deep Residual Learning for Image Recognition},
  author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  year={2015},
  eprint={1512.03385},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/1512.03385},
}

@misc{wightman2021resnetstrikesbackimproved,
  title={ResNet strikes back: An improved training procedure in timm},
  author={Ross Wightman and Hugo Touvron and Hervé Jégou},
  year={2021},
  eprint={2110.00476},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2110.00476},
}
```

**APA:**

He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep residual learning for image recognition*. arXiv. https://arxiv.org/abs/1512.03385

Wightman, R., Touvron, H., & Jégou, H. (2021). *ResNet strikes back: An improved training procedure in timm*. arXiv. https://arxiv.org/abs/2110.00476

## Glossary

- **Linear probe:** a linear classifier trained on the features of a frozen network; it measures how linearly separable the classes are in these features.
- **Brain-Score:** a platform that scores models by the similarity of their activations and behavior to those of primates on visual tasks.
- **Region commitment:** the layer whose activations Brain-Score compares to the recordings from a brain region.

## Model Card Authors

Niels Thiemann

## Model Card Contact

[niels-c](https://huggingface.co/niels-c) on Hugging Face
