README for unfreeze_layers
Overview

This folder contains different training configurations for freezing/unfreezing layers in the model.
Note: In all cases, the readout layers are always unfrozen.

Folder Descriptions

unfreeze_readout
Only the readout layers are unfrozen. All GNN backbone layers are frozen.

unfreeze_readout_backbone_L3
Readout layers and Layer 3 of the backbone GNN are unfrozen. Layers 1 and 2 of the backbone are frozen.

unfreeze_readout_backbone_L2_L3
Readout layers and Layers 2 & 3 of the backbone GNN are unfrozen. Layer 1 of the backbone is frozen.
