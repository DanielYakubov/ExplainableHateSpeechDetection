# ExplainableHateSpeechDetection

### REWORK PLANNED FOR MARCH 2025

This repo hosts code for the Constituent Rationale Explainability Framework (C-REF). This Framework was developed in order to attempt to solve the issue of hate speech classification systems not providing human rationales for their responses. More details on the motivation can be seen in the PDF paper also hosted in this repo. 

In order to use this Repo, the classifier component of the model must be fine-tuned on span consituents. For the HateXplain dataset used in the paper, this dataset is already generated, for similar datasets, one would have to use the preprocessing.py script under helpers/. Once the model is trained, evaluator.py and the inferencer.py scripts can be used.

This repo hosts hateful and vile language data, please peruse at your own risk.
