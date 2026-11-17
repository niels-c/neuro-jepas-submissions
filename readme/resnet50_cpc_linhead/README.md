---
license: mit
---

# ResNet-50 CPC with an ImageNet-1k linear head

A ResNet-50 pretrained with Contrastive Predictive Coding (CPC v2), with a linear classification head for the 1000 ImageNet-1k classes, packaged for Brain-Score.

## Model Details

### Model Description

The ResNet-50 backbone was trained self-supervised with CPC v2 on ImageNet-1k in the neuro-jepas project (experiment `resnet50_cpc_in1k`; method adapted from [rschwarz15/CPCV2-PyTorch](https://github.com/rschwarz15/CPCV2-PyTorch)). As in CPC v2, the backbone uses layer normalization (`timm` GroupNorm with one group) instead of batch normalization, because batch statistics would let the patches of an image share information. The backbone was then frozen and a linear head on the globally average-pooled output of the last stage (2048 dimensions) was trained on ImageNet-1k (linear probe). The head provides the 1000 ImageNet logits that the behavioral benchmarks of Brain-Score require.

- **Developed by:** Niels Thiemann (backbone training, linear head, packaging); method by van den Oord et al. (2018) and Hénaff et al. (2019)
- **Shared by:** [niels-c](https://huggingface.co/niels-c)
- **Model type:** Image classifier: self-supervised convolutional ResNet-50 backbone (layer normalization) with a linear head (1000 ImageNet-1k classes)
- **License:** MIT

### Model Sources

- **Repository:** [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions) (Brain-Score plugin), [neuro-jepas](https://github.com/niels-c/neuro-jepas) (training code)
- **Paper:** https://arxiv.org/abs/1807.03748 (CPC), https://arxiv.org/abs/1905.09272 (CPC v2)

## Uses

### Direct Use

Classifying images into the 1000 ImageNet-1k classes, and extracting the activations of its layers for comparison with neural and behavioral data of the primate visual system.

### Downstream Use

The model was built as a submission to [Brain-Score](https://www.brain-score.org/vision/), where it is scored as the plugin `resnet50_cpc_linhead` on neural (V1, V2, V4, IT) and behavioral benchmarks. It is part of the neuro-jepas project, which compares joint-embedding predictive architectures (JEPAs) and other self-supervised models on their similarity to the brain.

### Out-of-Scope Use

- Classification beyond a rough baseline: with 27.5% top-1 accuracy on ImageNet-1k, the model is not a usable classifier.
- Production or safety-critical image classification. The head is a linear probe on a frozen backbone and was not tuned for robustness or calibration.
- Images far from the ImageNet-1k distribution (e.g., medical, satellite or synthetic images) and labels outside the 1000 ImageNet-1k classes.
- Any decision about people. ImageNet-1k classes are not designed to describe people.

## Bias, Risks, and Limitations

- The model inherits the biases of ImageNet-1k, e.g., its class selection, the geographic and demographic skew of its photos and its label noise.
- Only the linear head was trained on labels; the backbone features were not adapted to classification, so accuracy is lower than for a fine-tuned model.
- The backbone was pretrained on 56 × 56 patches (a 7 × 7 grid of half-overlapping patches of 224 × 224 crops), but the head and Brain-Score apply it to whole 224 × 224 images.
- The backbone is much weaker than the CPC v2 ResNet-50 of Hénaff et al. (2019) (63.8% top-1 with a linear probe), whose training setup was much larger; the model reproduces the method, not the published results.
- The ResNet-50 uses layer normalization, so it cannot be loaded into `torchvision.models.resnet50` and differs in architecture from the other ResNet-50 submissions of neuro-jepas, including the untrained baseline.
- The model expects 224 × 224 RGB images. The probe was evaluated with a resize to 256 pixels and a 224 center crop, whereas Brain-Score resizes the whole image to 224 × 224, which changes the aspect ratio.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Use the code below, which builds the backbone with layer normalization. When comparing models on Brain-Score, also compare with the untrained baseline of the same architecture ([resnet50_untrained_linhead](https://huggingface.co/niels-c/resnet50_untrained_linhead)), keeping in mind that it uses batch normalization.

## How to Get Started with the Model

Requirements: `pip install torch torchvision "timm>=1.0.30" huggingface_hub safetensors`

```python
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torchvision.transforms import v2

# ResNet-50 with layer normalization (GroupNorm with one group), as in CPC v2
model = timm.create_model("resnet50", num_classes=0, norm_layer="groupnorm1")
model.fc = nn.Linear(model.head_hidden_size, 1000)
model.load_state_dict(
    load_file(
        hf_hub_download("niels-c/resnet50_cpc_linhead", "resnet50_cpc_linhead.safetensors")
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

To score the model on Brain-Score, use the plugin `resnet50_cpc_linhead` from [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions/tree/main/models/resnet50_cpc_linhead), which loads this checkpoint:

```python
import brainscore_vision

model = brainscore_vision.load_model("resnet50_cpc_linhead")
```

## Training Details

### Training Data

Backbone and linear head: the training split of ImageNet-1k (ILSVRC 2012; 1,281,167 images, 1000 classes) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k). The backbone was trained without the labels.

### Training Procedure

Both stages used the trainer of [neuro-jepas](https://github.com/niels-c/neuro-jepas).

**Backbone (experiment `resnet50_cpc_in1k`):** CPC v2. Each image is cut into a 7 × 7 grid of half-overlapping 56 × 56 patches, each augmented independently and encoded separately by the ResNet-50. A PixelCNN-style context network summarizes each patch's row and the rows above, and linear predictors predict the encodings of the patches up to 5 rows below, in 4 directions (the grid rotated by 90 degrees per direction), contrasted with 128 negative encodings from other patches of the batch (InfoNCE loss). Only the ResNet-50 was exported.

- **Epochs:** at most 100, with early stopping (patience 10) on the validation loss
- **Batch size:** 128
- **Optimizer:** AdamW, weight decay 0
- **Learning rate:** 2e-4 peak after a linear warmup of 2 epochs, cosine decay to 1e-6

**Linear head (experiment `resnet50_cpc_linhead`):** the backbone was frozen and only the linear head was trained with the cross-entropy loss.

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
| Top-1 accuracy | 27.45% |
| Top-5 accuracy | 51.13% |
| Cross-entropy | 3.616 |

#### Summary

The frozen CPC features reach 27.5% top-1 accuracy with a linear head on ImageNet-1k, well above the untrained ResNet-50 (2.9%) but below the other self-supervised ResNet-50s of neuro-jepas.

## Technical Specifications

### Model Architecture and Objective

ResNet-50 as in `timm.create_model("resnet50", norm_layer="groupnorm1")` (bottleneck blocks in 4 stages of 3, 4, 6 and 3 blocks, every normalization layer a GroupNorm with one group, i.e., layer normalization over channels and positions), whose layer `fc` is the linear head on the globally average-pooled 2048-dimensional features.

Backbone: self-supervised CPC v2 objective (contrastive prediction of patch encodings). Head: cross-entropy on ImageNet-1k labels.

Brain-Score commitments of the plugin `resnet50_cpc_linhead`: the recorded layers are `backbone.conv1`, `backbone.layer1`, `backbone.layer2`, `backbone.layer3`, `backbone.layer4`, `backbone.fc`, and `backbone.fc` is the behavioral readout. The regions are committed to these layers:

| Region | Layer |
|---|---|
| V1 | `backbone.layer1` |
| V2 | `backbone.layer2` |
| V4 | `backbone.layer3` |
| IT | `backbone.layer4` |

### Compute Infrastructure

#### Software

PyTorch, `timm` >= 1.0.30, safetensors, huggingface_hub; backbone and head were trained with the neuro-jepas trainer.

## Citation

Please cite the papers of the method:

**BibTeX:**

```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
  title={Representation Learning with Contrastive Predictive Coding},
  author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
  year={2019},
  eprint={1807.03748},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/1807.03748},
}

@misc{henaff2020dataefficientimagerecognitioncontrastive,
  title={Data-Efficient Image Recognition with Contrastive Predictive Coding},
  author={Olivier J. Hénaff and Aravind Srinivas and Jeffrey De Fauw and Ali Razavi and Carl Doersch and S. M. Ali Eslami and Aaron van den Oord},
  year={2020},
  eprint={1905.09272},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/1905.09272},
}
```

**APA:**

van den Oord, A., Li, Y., & Vinyals, O. (2019). *Representation learning with contrastive predictive coding*. arXiv. https://arxiv.org/abs/1807.03748

Hénaff, O. J., Srinivas, A., De Fauw, J., Razavi, A., Doersch, C., Eslami, S. M. A., & van den Oord, A. (2020). *Data-efficient image recognition with contrastive predictive coding*. arXiv. https://arxiv.org/abs/1905.09272

## Glossary

- **Linear probe:** a linear classifier trained on the features of a frozen network; it measures how linearly separable the classes are in these features.
- **Brain-Score:** a platform that scores models by the similarity of their activations and behavior to those of primates on visual tasks.
- **Region commitment:** the layer whose activations Brain-Score compares to the recordings from a brain region.

## Model Card Authors

Niels Thiemann

## Model Card Contact

[niels-c](https://huggingface.co/niels-c) on Hugging Face
