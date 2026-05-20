# Small ImageNet Sample

This folder contains a tiny subset (200 images) of the ImageNet validation set, organized in the usual
`root/<class_name>/<image>.jpg` structure.  It is *not* representative of the full dataset, but is
sufficient to demonstrate the evaluation pipeline.  The class names correspond to the synset IDs
listed in `imagenet_sample.json`.

To use the dataset with the script, simply point `ID_DATASET_DIR` in `evaluate.py` to this folder.