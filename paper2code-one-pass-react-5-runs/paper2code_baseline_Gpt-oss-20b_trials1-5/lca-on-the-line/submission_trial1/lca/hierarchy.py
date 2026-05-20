"""
Utility to map CIFAR‑10 class indices to WordNet synsets.
"""

from typing import Dict, List
from nltk.corpus import wordnet as wn


class SynsetMapper:
    """
    Maps between integer class indices and WordNet synsets.
    Provides quick lookup of depth and hypernym paths.
    """

    def __init__(self, class_to_wnid: Dict[int, str]):
        """
        Parameters
        ----------
        class_to_wnid : dict[int, str]
            Mapping from CIFAR‑10 class index to WordNet synset ID.
        """
        self.class_to_wnid = class_to_wnid
        self.wnid_to_synset = {
            idx: wn.synset(wnid) for idx, wnid in class_to_wnid.items()
        }
        self.synset_to_depth = {
            idx: syn.depth() for idx, syn in self.wnid_to_synset.items()
        }

    def get_synset(self, class_idx: int):
        """Return the WordNet synset for the given class index."""
        return self.wnid_to_synset[class_idx]

    def get_depth(self, class_idx: int) -> int:
        """Return the depth of the synset for the given class index."""
        return self.synset_to_depth[class_idx]