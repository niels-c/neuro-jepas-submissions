---
license: mit
---

# ResNet-50 CNN-JEPA with an ImageNet-1k linear head

A ResNet-50 pretrained with CNN-JEPA (a Joint-Embedding Predictive Architecture for convolutional networks), with a linear classification head for the 1000 ImageNet-1k classes, packaged for Brain-Score.

## Model Details

### Model Description

The ResNet-50 backbone was trained self-supervised with CNN-JEPA on ImageNet-1k for 200 epochs in the neuro-jepas project (experiment `resnet50_ijepacnn_in1k200`; method adapted from [kaland313/CNN-JEPA](https://github.com/kaland313/CNN-JEPA)). CNN-JEPA carries I-JEPA over to CNNs: a sparse copy of the encoder sees only the visible context of a masked image, a convolutional predictor fills in the masked positions of its feature map, and the predictions are regressed onto the features of the full image from an exponential moving average (EMA) of the encoder. The EMA target encoder was exported, frozen, and a linear head on its globally average-pooled output of the last stage (2048 dimensions) was trained on ImageNet-1k (linear probe). The head provides the 1000 ImageNet logits that the behavioral benchmarks of Brain-Score require.

- **Developed by:** Niels Thiemann (backbone training, linear head, packaging); method by Kalapos & Gyires-Tóth (2024)
- **Shared by:** [niels-c](https://huggingface.co/niels-c)
- **Model type:** Image classifier: self-supervised convolutional ResNet-50 backbone with a linear head (1000 ImageNet-1k classes)
- **License:** MIT

### Model Sources

- **Repository:** [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions) (Brain-Score plugin), [neuro-jepas](https://github.com/niels-c/neuro-jepas) (training code)
- **Paper:** https://arxiv.org/abs/2408.07514

## Uses

### Direct Use

Classifying images into the 1000 ImageNet-1k classes, and extracting the activations of its layers for comparison with neural and behavioral data of the primate visual system.

### Downstream Use

The model was built as a submission to [Brain-Score](https://www.brain-score.org/vision/), where it is scored as the plugin `resnet50_ijepacnn_linhead` on neural (V1, V2, V4, IT) and behavioral benchmarks. It is part of the neuro-jepas project, which compares joint-embedding predictive architectures (JEPAs) and other self-supervised models on their similarity to the brain.

### Out-of-Scope Use

- Production or safety-critical image classification. The head is a linear probe on a frozen backbone and was not tuned for robustness or calibration.
- Images far from the ImageNet-1k distribution (e.g., medical, satellite or synthetic images) and labels outside the 1000 ImageNet-1k classes.
- Any decision about people. ImageNet-1k classes are not designed to describe people.

## Bias, Risks, and Limitations

- The model inherits the biases of ImageNet-1k, e.g., its class selection, the geographic and demographic skew of its photos and its label noise.
- Only the linear head was trained on labels; the backbone features were not adapted to classification, so accuracy is lower than for a fine-tuned model.
- The backbone is one pretraining run (seed 0) of 200 epochs on a single GPU; it reproduces the method, not the published results.
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
            "niels-c/resnet50_ijepacnn_linhead", "resnet50_ijepacnn_linhead.safetensors"
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

To score the model on Brain-Score, use the plugin `resnet50_ijepacnn_linhead` from [neuro-jepas-submissions](https://github.com/niels-c/neuro-jepas-submissions/tree/main/models/resnet50_ijepacnn_linhead), which loads this checkpoint:

```python
import brainscore_vision

model = brainscore_vision.load_model("resnet50_ijepacnn_linhead")
```

## Training Details

### Training Data

Backbone and linear head: the training split of ImageNet-1k (ILSVRC 2012; 1,281,167 images, 1000 classes) from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k). The backbone was trained without the labels.

### Training Procedure

Both stages used the trainer of [neuro-jepas](https://github.com/niels-c/neuro-jepas), with the ImageNet-1k configuration of the CNN-JEPA paper for the backbone.

**Backbone (experiment `resnet50_ijepacnn_in1k200`):** CNN-JEPA. Masks are drawn on the grid of the final feature map, per batch: multi-block masks (4 target blocks of 15–20% of the grid each, aspect ratio 0.75–1.5, hidden from the context) with probability 0.75, otherwise random masks that hide 60% of the positions. The predictor is 3 depthwise separable convolutions (kernel size 3) with batch normalization and ReLU. The EMA momentum of the target encoder rises from 0.996 to 1.0.

- **Epochs:** 200
- **Batch size:** 512 (the paper's 4 GPUs × 128, on one GPU)
- **Optimizer:** AdamW, weight decay 0.01
- **Learning rate:** 1e-2 peak after a linear warmup of 10 epochs, cosine decay to 1e-5

**Linear head (experiment `resnet50_ijepacnn200_linhead`):** the backbone was frozen (batch normalization statistics included) and only the linear head was trained with the cross-entropy loss.

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
| Top-1 accuracy | 58.42% |
| Top-5 accuracy | 82.04% |
| Cross-entropy | 1.762 |

#### Summary

The frozen CNN-JEPA features reach 58.4% top-1 accuracy with a linear head on ImageNet-1k.

## Technical Specifications

### Model Architecture and Objective

ResNet-50 as in `torchvision.models.resnet50` (bottleneck blocks in 4 stages of 3, 4, 6 and 3 blocks), whose layer `fc` is the linear head on the globally average-pooled 2048-dimensional features. It was trained as `timm.create_model("resnet50")`, whose parameter names are the same.

Backbone: self-supervised CNN-JEPA objective (regression of masked feature-map positions onto EMA target features). Head: cross-entropy on ImageNet-1k labels.

Brain-Score commitments of the plugin `resnet50_ijepacnn_linhead`: the recorded layers are `backbone.conv1`, `backbone.layer1`, `backbone.layer2`, `backbone.layer3`, `backbone.layer4`, `backbone.fc`, and `backbone.fc` is the behavioral readout. The regions are committed to these layers:

| Region | Layer |
|---|---|
| V1 | `backbone.layer1` |
| V2 | `backbone.layer2` |
| V4 | `backbone.layer3` |
| IT | `backbone.layer4` |

### Compute Infrastructure

#### Software

PyTorch, torchvision (inference), `timm` (training), safetensors, huggingface_hub; backbone and head were trained with the neuro-jepas trainer.

## Citation

Please cite the paper of the method:

**BibTeX:**

```bibtex
@inproceedings{Kalapos_2024,
  title={CNN-JEPA: Self-Supervised Pretraining Convolutional Neural Networks Using Joint Embedding Predictive Architecture},
  url={http://dx.doi.org/10.1109/ICMLA61862.2024.00169},
  DOI={10.1109/icmla61862.2024.00169},
  booktitle={2024 International Conference on Machine Learning and Applications (ICMLA)},
  publisher={IEEE},
  author={Kalapos, András and Gyires-Tóth, Bálint},
  year={2024},
  month=Dec,
  pages={1111--1114},
}
```

**APA:**

Kalapos, A., & Gyires-Tóth, B. (2024). CNN-JEPA: Self-supervised pretraining convolutional neural networks using joint embedding predictive architecture. In *2024 International Conference on Machine Learning and Applications (ICMLA)* (pp. 1111–1114). IEEE. https://doi.org/10.1109/ICMLA61862.2024.00169

## Glossary

- **Linear probe:** a linear classifier trained on the features of a frozen network; it measures how linearly separable the classes are in these features.
- **JEPA:** joint-embedding predictive architecture; a self-supervised model that predicts the representations, not the pixels, of masked parts of its input.
- **Brain-Score:** a platform that scores models by the similarity of their activations and behavior to those of primates on visual tasks.
- **Region commitment:** the layer whose activations Brain-Score compares to the recordings from a brain region.

## Model Card Authors

Niels Thiemann

## Model Card Contact

[niels-c](https://huggingface.co/niels-c) on Hugging Face
