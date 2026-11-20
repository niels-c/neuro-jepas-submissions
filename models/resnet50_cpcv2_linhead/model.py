"""
Model definition of a ResNet-50 trained with CPC v2 for submission to Brain-Score.

As in the evaluation of CPC v2, the ResNet-50 encodes patches of the size it was trained on: each 224x224 image is cut
into a 5x5 grid of 80x80 patches, 36 pixels apart as in pretraining, which are encoded separately. The recorded layers
are the stage outputs of all patches of an image, arranged as a mosaic in the layout of the patches; the linear head
classifies the average of the pooled patch features.

Checkpoint from https://huggingface.co/niels-c/resnet50_cpcv2_linhead.
"""

import functools
from collections import OrderedDict

import timm
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

IMAGE_SIZE = 224
GRID_SIZE = 5
PATCH_SIZE = 80
NUM_CLASSES = 1000
STAGES = ("conv1", "layer1", "layer2", "layer3", "layer4")
REPO_ID = "niels-c/resnet50_cpcv2_linhead"
FILENAME = "resnet50_cpcv2_linhead.safetensors"


class PatchedResNet(nn.Module):
    """
    timm ResNet-50 with layer norm applied to a grid of patches of each image, with a linear head (fc) on the average of
    the pooled patch features, returning logits.

    The modules of the ResNet are its direct children, so the weights have the keys of the ResNet. The output of each
    stage also passes, as a mosaic with one row per image, through the identity grid.<stage>, which Brain-Score records.
    It is the architecture "patched_resnet" of the training runs.
    """

    def __init__(self):
        super().__init__()
        resnet = timm.create_model("resnet50", num_classes=0, norm_layer="groupnorm1")
        for name, module in resnet.named_children():
            self.add_module(name, module)
        self.fc = nn.Linear(resnet.head_hidden_size, NUM_CLASSES)
        self.grid = nn.ModuleDict({stage: nn.Identity() for stage in STAGES})

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        stride = (images.shape[-1] - PATCH_SIZE) // (GRID_SIZE - 1)
        patches = images.unfold(-2, PATCH_SIZE, stride).unfold(-2, PATCH_SIZE, stride)
        x = patches.movedim(1, 3).flatten(0, 2)  # (B * G * G, 3, P, P), row-major over the grid
        x = self.conv1(x)
        self.grid.conv1(self.to_mosaic(x))
        x = self.maxpool(self.act1(self.bn1(x)))
        for stage in STAGES[1:]:
            x = getattr(self, stage)(x)
            self.grid[stage](self.to_mosaic(x))
        features = self.global_pool(x).view(len(images), GRID_SIZE * GRID_SIZE, -1).mean(dim=1)
        return self.fc(features)

    @staticmethod
    def to_mosaic(features: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = features.shape
        features = features.view(-1, GRID_SIZE, GRID_SIZE, channels, height, width)
        return features.permute(0, 3, 1, 4, 2, 5).reshape(-1, channels, GRID_SIZE * height, GRID_SIZE * width)


def load_model() -> nn.Module:
    """
    Load the network from its checkpoint on the Hugging Face Hub.

    Returns:
        nn.Module: A Sequential of "backbone", the patched ResNet-50 whose layer fc is the linear head (logits), in
            eval mode.
    """

    weights = load_file(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
    backbone = PatchedResNet()
    backbone.load_state_dict(weights)
    return nn.Sequential(OrderedDict(backbone=backbone)).eval()


def get_model(name):
    assert name == "resnet50_cpcv2_linhead"
    model = load_model()  # head at backbone.fc, logits
    # The weights were trained with the ImageNet mean and std, the default of load_preprocess_images.
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier="resnet50_cpcv2_linhead", model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == "resnet50_cpcv2_linhead"
    # Stage outputs of all patches as mosaics.
    return [f"backbone.grid.{stage}" for stage in STAGES] + ["backbone.fc"]


def get_bibtex(model_identifier):
    return """@inproceedings{henaff2020dataefficient,
      title={Data-Efficient Image Recognition with Contrastive Predictive Coding},
      author={Olivier J. H{\\'e}naff and Aravind Srinivas and Jeffrey De Fauw and Ali Razavi and Carl Doersch and S. M. Ali Eslami and Aaron van den Oord},
      booktitle={Proceedings of the 37th International Conference on Machine Learning},
      pages={4182--4192},
      year={2020},
      publisher={PMLR},
}"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
