# ORION

> Open-source Real-time Inference of Open-ended Notes (ORION) on artworks.

This repository is code for kekbot's affective captioning for short generative open-ended art criticism,
via ArtEmis speakers and BLIP, referred to as the ORION method.

## How to use:

1. **Get access to the ArtEmis dataset and speaker models**  
   In order to respect the original authors' wishes, I will not be publishing the dataset (and its speaker models) as-is along with this repository, I suggest visiting https://artemisdataset.org to formally accept the ToS first.
2. **Preprocess the data**  
   Clone [**au-artemis**](https://github.com/spuuntries/au-artemis), then do in the root of this repo:

   > Replace "python" with python3 if it's not bound to python 3, replace the [INSERTSOMETHING] with the paths.

   ` python [INSERTAU-ARTEMISPATH]/artemis/scripts/preprocess_artemis_data.py -save-out-dir models-data/preprocessed-data -raw-artemis-data-csv [INSETARTEMISDATASETPATH]/artemis_dataset_release_v0.csv --preprocess-for-deep-nets True`

3.
