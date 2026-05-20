# Data Directory

This directory contains the data files used by the reproduction scripts.

## Files

- `sample_predictions.npy`: A numpy array containing simulated model predictions.
- `sample_labels.npy`: A numpy array containing the true labels.
- `wordnet_hierarchy.pkl`: A pickle file containing the WordNet class hierarchy.

## Notes

In a real implementation, these files would be downloaded from the paper's GitHub repository. For this reproduction, we use simulated data to replicate the 75 models from the paper.

The files are used by the `reproduce.sh` script to calculate the LCA distance and generate the results.

The data is structured to be compatible with the `lca_calculator.py` script.

## Sources

The data is simulated based on the paper's methodology.

The WordNet hierarchy is based on the WordNet 3.1 database.

## Contact

For questions or issues, please contact the reproduction maintainer: reproduction@lca-repro.com