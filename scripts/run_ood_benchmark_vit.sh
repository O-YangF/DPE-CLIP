#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main_dpe.py   --config configs \
                                            --wandb-log \
                                            --datasets A \
                                            --backbone ViT-B/16 \
                                            # --coop