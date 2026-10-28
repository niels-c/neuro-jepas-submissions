---
license: mit
---

# ResNet-50 SimSiam with an ImageNet-1k linear head

A ResNet-50 pretrained with SimSiam, with a linear classification head for the 1000 ImageNet-1k classes, packaged for Brain-Score.

## Model Details

### Model Description

The ResNet-50 backbone was trained self-supervised with SimSiam on ImageNet-1k (see [facebookresearch/simsiam](https://github.com/facebookresearch/simsiam)). Backbone and linear head (on the globally average-pooled output of the last stage (2048 dimensions)) are the checkpoint released in [facebookresearch/simsiam](https://github.com/facebookresearch/simsiam), converted to the parameter names of `torchvision.models.resnet50` and saved as safetensors. The head was not retrained. The head provides the 1000 ImageNet logits that the behavioral benchmarks of Brain-Score require.

- **Developed by:** backbone by Meta AI (Chen & He, 2020); packaging by Niels Thiemann
- **Shared by:** [niels-c](https://huggingface.co/niels-c)
- **Model type:** Image classifier: self-supervised convolutional ResNet-50 backbone with a linear head (1000 ImageNet-1k classes)
- **License:** MIT; the backbone weights are also subject to the license of their source repository
- **Finetuned from model:** [facebookresearch/simsiam](https://github.com/facebookresearch/simsiam) (frozen backbone)

### Model Sources

- **Repository:** [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions) (Brain-Score plugin), [neuro-jepas](https://github.com/niels-c/neuro-jepas) (training code)
- **Paper:** https://arxiv.org/abs/2011.10566

## Uses

### Direct Use

Classifying images into the 1000 ImageNet-1k classes, and extracting the activations of its layers for comparison with neural and behavioral data of the primate visual system.

### Downstream Use

The model was built as a submission to [Brain-Score](https://www.brain-score.org/vision/), where it is scored as the plugin `resnet50_simsiam_linhead` on neural (V1, V2, V4, IT) and behavioral benchmarks. It is part of the neuro-jepas project, which compares joint-embedding predictive architectures (JEPAs) and other self-supervised models on their similarity to the brain.

### Out-of-Scope Use

- Production or safety-critical image classification. The head is a linear probe on a frozen backbone and was not tuned for robustness or calibration.
- Images far from the ImageNet-1k distribution (e.g., medical, satellite or synthetic images) and labels outside the 1000 ImageNet-1k classes.
- Any decision about people. ImageNet-1k classes are not designed to describe people.

## Bias, Risks, and Limitations

- The model inherits the biases of ImageNet-1k, e.g., its class selection, the geographic and demographic skew of its photos and its label noise.
- Only the linear head was trained in the source repository; the backbone features were not adapted to classification, so accuracy is lower than for a fine-tuned model.
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
model.load_state_dict(load_file(hf_hub_download("niels-c/resnet50_simsiam_linhead", "resnet50_simsiam_linhead.safetensors")))
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

To score the model on Brain-Score, use the plugin `resnet50_simsiam_linhead` from [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions/tree/main/models/resnet50_simsiam_linhead), which pins the revision of this checkpoint:

```python
import brainscore_vision

model = brainscore_vision.load_model("resnet50_simsiam_linhead")
```

## Training Details

### Training Data

ImageNet-1k (ILSVRC 2012). Backbone: self-supervised with SimSiam on ImageNet-1k (see [facebookresearch/simsiam](https://github.com/facebookresearch/simsiam)). The linear head comes with the source checkpoint; see the source repository for its training data and procedure.

### Training Procedure

Not trained in this project. The checkpoint of [facebookresearch/simsiam](https://github.com/facebookresearch/simsiam) was converted to the parameter names of `torchvision.models.resnet50` and saved as safetensors.

#### Speeds, Sizes, Times

- **Parameters:** 25,557,032 (backbone 23,508,032, head 2,049,000)
- **Checkpoint:** 103 MB, float32 safetensors

## Technical Specifications

### Model Architecture and Objective

ResNet-50 as in `torchvision.models.resnet50` (bottleneck blocks in 4 stages of 3, 4, 6 and 3 blocks), whose layer `fc` is the linear head on the globally average-pooled 2048-dimensional features.

Backbone: self-supervised SimSiam objective. Head: cross-entropy on ImageNet-1k labels.

Brain-Score commitments of the plugin `resnet50_simsiam_linhead`: the recorded layers are `backbone.conv1`, `backbone.layer1`, `backbone.layer2`, `backbone.layer3`, `backbone.layer4`, `backbone.fc`, and `backbone.fc` is the behavioral readout. The regions are committed to these layers:

| Region | Layer |
|---|---|
| V1 | `backbone.layer1` |
| V2 | `backbone.layer2` |
| V4 | `backbone.layer3` |
| IT | `backbone.layer4` |

### Compute Infrastructure

#### Software

PyTorch, torchvision (inference), `timm` (training), safetensors, huggingface_hub.

## Citation

Please cite the paper of the backbone:

**BibTeX:**

```bibtex
@misc{chen2020exploringsimplesiameserepresentation,
  title={Exploring Simple Siamese Representation Learning},
  author={Xinlei Chen and Kaiming He},
  year={2020},
  eprint={2011.10566},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2011.10566},
}
```

**APA:**

Chen, X., & He, K. (2020). *Exploring simple Siamese representation learning*. arXiv. https://arxiv.org/abs/2011.10566

## Glossary

- **Linear probe:** a linear classifier trained on the features of a frozen network; it measures how linearly separable the classes are in these features.
- **Brain-Score:** a platform that scores models by the similarity of their activations and behavior to those of primates on visual tasks.
- **Region commitment:** the layer whose activations Brain-Score compares to the recordings from a brain region.

## Model Card Authors

Niels Thiemann

## Model Card Contact

[niels-c](https://huggingface.co/niels-c) on Hugging Face
