# CBE-ESD
Concept Based Explanation for Error Slice Discovery

How to use the code?
1. Train a CBM using train.py
2. Find the set of concepts with the highest expected change in the target prediction score (i.e., the concepts that are most likely contributing to the model's error using analyze_erroneous_concepts.py
3. Fit a GMM model on the pre-trained CBM using train_gmm.py
4. Run quantitative and qualitative evaluation using slice/test_concept_aware.py 
