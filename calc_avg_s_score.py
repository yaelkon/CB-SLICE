import json
import numpy as np


concept_analysis_summary = json.load(open('experiments/cbm/MetaShiftCatDog/ValDistAsTrain/20251119-124858_CBM:ValDistAsTrain_Sequential_lr:0.1_momentum:0.1_wd:0.00001_Task/concept_analysis/comprehensive_analysis_summary.json'))

top_k_concepts_by_s_score = concept_analysis_summary['top_k_concepts_by_s_score']

for class_idx, class_data in top_k_concepts_by_s_score.items():
    print(f"Class {class_idx}:")
    avg_s_score = 0
    for concept_idx, s_score in class_data.items():
        avg_s_score += s_score
    avg_s_score /= len(class_data)
    # Find the location of the concept which its value is greater than avg_s_score
    concepts_above_avg = [concept_idx for concept_idx, s_score in class_data.items() if s_score > avg_s_score]
    print(f"  Concepts above average S score: {concepts_above_avg}")
    print(f"  Average S score: {avg_s_score}")