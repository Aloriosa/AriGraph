"""
Utilities for mapping ImageNet class indices to WordNet synsets.
"""

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
    # The dataset.classes list is sorted alphabetically by folder name.
    # The ImageNet class index is the order of the folders.
    class_names = dataset.classes  # e.g. ['n01440764', 'n01443537', ...]
    synsets = []

    for name in class_names:
        try:
            synset = wn.synset_from_pos_and_offset('n', int(name[1:]))
        except Exception:
            # fallback: try by name
            synset = wn.synset(name + ".n.01")
        synsets.append(synset)

    return synsets