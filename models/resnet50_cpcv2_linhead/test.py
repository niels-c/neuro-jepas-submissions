"""
Test script for the ResNet-50 CPC v2 linear head model.
"""

import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("resnet50_cpcv2_linhead")
    assert model.identifier == "resnet50_cpcv2_linhead"
