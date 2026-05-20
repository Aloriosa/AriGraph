"""
Deprecated copy of imagenet_utils.py – kept for backward compatibility.
"""

import os
import torch
import torchvision
import nltk
from nltk.corpus import wordnet as wn

def get_imagenet_class_mappings(dataset):
    """
    Build a list mapping each class index to a WordNet synset.

    Parameters
    ----------
    dataset : torchvision.datasets.ImageFolder
        The ImageNet validation dataset.

    Returns
    -------
    list[wn.synset]
        List where index i corresponds to the synset for class i.
    """
    # The dataset.classes list is sorted alphabetically by folder name
    # (e.g. ['n01440764', 'n01443537', ...]).
    class_names = dataset.classes
    synsets = []

    for name in class_names:
        # name is of the form 'n01440764'
        try:
            offset = int(name[1:])  # strip the 'n' prefix
            synset = wn.synset_from_pos_and_offset('n', offset)
        except Exception:
            # fallback: try by the synset name
            synset = wn.synset(name + ".n.01")
        synsets.append(synset)

    return synsets