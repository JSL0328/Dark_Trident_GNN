#!/bin/bash
apptainer exec --nv -B /vols \
  /vols/sbn/uboone/ll4420/dark_tridents_wspace/larcv2.sif \
  bash -c "source /vols/sbn/uboone/ll4420/dark_tridents_wspace/venv/bin/activate && \
           source /vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-CNN/setup_larcv2_dm.sh && \
           python3 /vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/train_gnn_transformer.py"
