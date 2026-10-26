# Neuro JEPAs submission repository
This repository contains the plugins submitted to the brainscore-vision repository as part of the neuro-jepas project.

Each plugin downloads its checkpoint (safetensors) from the Hugging Face Hub repository `niels-c/<model identifier>`, as set by `REPO_ID` and `FILENAME` in its `model.py`.


**Disclaimer**
Checkpoints for the backbone were partially obtained from other repositories. These are linked in the respective `model.py` files.