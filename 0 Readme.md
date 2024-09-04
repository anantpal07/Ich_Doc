# RSNA Intracranial Hemorrhage Detection Documentation
## 1. Overview
### 1.1. Description
Intracranial hemorrhage is a critical condition where bleeding occurs within the skull, often leading to life-threatening complications. Prompt diagnosis and treatment are vital. This challenge aims to develop an algorithm that can accurately detect acute intracranial hemorrhage and its subtypes using a rich dataset of medical images provided by the Radiological Society of North America (RSNA®), in collaboration with the American Society of Neuroradiology and MD.ai.

### 1.2. Objectives
Detect Presence of Hemorrhage: Identify whether an intracranial hemorrhage is present in a given medical image.
Classify Hemorrhage Subtypes: Determine the specific subtype of hemorrhage if present.
### 1.3. Significance
Accurate and timely identification of hemorrhage subtypes can assist medical professionals in prioritizing and expediting patient care, potentially saving lives.

## 2. Dataset Description
### 2.1. Data Format
Training Data: Provided in stage_2_train.csv, containing image IDs and corresponding labels for five subtypes of hemorrhage and an additional label indicating the presence of any hemorrhage.
Test Data: Provided in stage_2_test.zip, containing the images for the current stage.
Sample Submission: Provided in stage_2_sample_submission.csv, demonstrating the required submission format.
### 2.2. Data Fields
Id: A unique identifier for each image, which includes an underscore _.
Label: The probability of whether a specific hemorrhage subtype (or any hemorrhage in the case of the "any" label) exists in the image.
### 2.3. File Descriptions
stage_2_train.csv: The training set with image IDs and corresponding labels.
stage_2_sample_submission.csv: A sample submission file in the correct format, containing IDs for the test set.
### 2.4. DICOM Images
All images are provided in DICOM format, which includes metadata such as PatientID, StudyInstanceUID, SeriesInstanceUID, etc.

## 3. Hemorrhage Types
### 3.1. Subtypes of Hemorrhage
Intraparenchymal Hemorrhage: Blood located completely within the brain tissue.
Intraventricular Hemorrhage: Blood within the brain's ventricular system.
Subarachnoid Hemorrhage: Blood in the subarachnoid space, the area between the brain and the thin tissues covering it.
Subdural Hemorrhage: Blood beneath the dura mater, the brain's outermost covering.
Epidural Hemorrhage: Blood between the skull and the dura mater.
### 3.2. Clinical Relevance
Hemorrhages vary in their severity, with larger hemorrhages typically being more dangerous. However, even small hemorrhages can be critical if they indicate underlying conditions such as aneurysms.
### 3.3. Imaging Characteristics
All acute hemorrhages appear dense (white) on CT scans. Radiologists determine the subtype based on the hemorrhage's location, shape, and proximity to critical brain structures.
## 4. Evaluation Criteria
###4.1. Evaluation Metric
Submissions are evaluated using a weighted multi-label logarithmic loss. The "any" label, indicating the presence of any hemorrhage, is given a higher weight than specific subtypes. The log loss is calculated for each predicted probability against the true label, and the overall loss is averaged across all samples.

### 4.2. Submission Requirements
For each image ID, a set of predicted probabilities must be submitted, with one row for each hemorrhage subtype. The submission file must contain a header and follow the format:
```
Id,Label
1_epidural,0
1_intraparenchymal,0
1_intraventricular,0
1_subarachnoid,0.6
1_subdural,0
1_any,0.9
2_epidural,0
...
```

### 4.3. Data Handling
The training data consists of multiple labels per image ID, with one row for each hemorrhage subtype, plus an additional label for the presence of any hemorrhage.
