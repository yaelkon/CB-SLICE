# CBE-ESD
Concept Based Explanation for Error Slice Discovery

How to use the code?
1. Install the packages and dependencies from the file `environment.yml`.
2. Download the datasets described in the manuscript and update the `data_path` variable in `./configs/data/data_defaults.yaml`.
3. For Weights & Biases support, set mode to 'online' and adjust entity in `./configs/config.yaml`.
4. Train a CBM using train.py with the desired configuration of dataset and model from the `./configs/` folder.
5. Find the set of concepts with the highest expected change in the target prediction score (i.e., the concepts that are most likely contributing to the model's error) using analyze_erroneous_concepts.py script.
6. Fit a GMM model on the pre-trained CBM using train_gmm.py script.
7. Run quantitative and qualitative evaluation using slice/test_concept_aware.py script.
