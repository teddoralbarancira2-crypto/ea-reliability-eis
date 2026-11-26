# -*- coding: utf-8 -*-
"""Paths for the analysis. Set them with environment variables or edit the defaults below.
 DATA_A_RAW  folder with dataset A raw exports ('OCV-EIS test' and 'Capacity test' subfolders); not public (data owner permission)
 DATA_B_DIR  folder with the dataset B HPPC/EIS .mat files; available on request
 WORK        folder for intermediate files (dataset A impedance cube)"""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_A_RAW = os.environ.get('EA_DATA_A_RAW', os.path.join(ROOT, 'data', 'dataset_a'))
DATA_B_DIR = os.environ.get('EA_DATA_B_DIR', os.path.join(ROOT, 'data', 'dataset_b'))
WORK = os.environ.get('EA_WORK', os.path.join(ROOT, 'work')); os.makedirs(WORK, exist_ok=True)
CUBE = os.path.join(WORK, 'cube.npz')
DATA_A_PREFIX = os.environ.get('EA_DATA_A_PREFIX', '')   # file-name prefix of the dataset A raw exports
AL = os.path.join(ROOT, 'arrhenius_labels')
PAPER = os.path.join(ROOT, 'paper')
