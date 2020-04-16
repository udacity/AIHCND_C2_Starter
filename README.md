# Pneumonia Detection from Chest X-Rays

## Project Overview

In this project, you will apply the skills that you have acquired in this 2D medical imaging course  to analyze data from the NIH Chest X-ray Dataset and train a CNN to classify a given chest x-ray for the presence or absence of pneumonia. This project will culminate in a model that can predict the presence of pneumonia with human radiologist-level accuracy that can be prepared for submission to the FDA for 510(k) clearance as software as a medical device. As part of the submission preparation, you will formally describe your model, the data that it was trained on, and a validation plan that meets FDA criteria.

You will be provided with the medical images with clinical labels for each image that were extracted from their accompanying radiology reports. 

The project will include access to a GPU for fast training of deep learning architecture, as well as access to 112,000 chest x-rays with disease labels  acquired from 30,000 patients.

## Pneumonia and X-Rays in the Wild

Chest X-ray exams are one of the most frequent and cost-effective types of medical imaging examinations. Deriving clinical diagnoses from chest X-rays can be challenging, however, even by skilled radiologists. 

When it comes to pneumonia, chest X-rays are the best available method for diagnosis. More than 1 million adults are hospitalized with pneumonia and around 50,000 die from the disease every
year in the US alone. The high prevalence of pneumonia makes it a good candidate for the development of a deep learning application for two reasons: 1) Data availability in a high enough quantity for training deep learning models for image classification 2) Opportunity for clinical aid by providing higher accuracy image reads of a difficult-to-diagnose disease and/or reduce clinical burnout by performing automated reads of very common scans. 

The diagnosis of pneumonia from chest X-rays is difficult for several reasons: 
1. The appearance of pneumonia in a chest X-ray can be very vague depending on the stage of the infection
2. Pneumonia often overlaps with other diagnoses
3. Pneumonia can mimic benign abnormalities

For these reasons, common methods of diagnostic validation performed in the clinical setting are to obtain sputum cultures to test for the presence of bacteria or viral bodies that cause pneumonia, reading the patient's clinical history and taking their demographic profile into account, and comparing a current image to prior chest X-rays for the same patient if they are available. 

## About the Dataset

The dataset provided to you for this project was curated by the NIH specifically to address the problem of a lack of large x-ray datasets with ground truth labels to be used in the creation of disease detection algorithms. 

There are 112,120 X-ray images with disease labels from 30,805 unique patients in this dataset.  The disease labels were created using Natural Language Processing (NLP) to mine the associated radiological reports. The labels include 14 common thoracic pathologies: 
- Atelectasis 
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia 

The biggest limitation of this dataset is that image labels were NLP-extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.

The original radiology reports are not publicly available but you can find more details on the labeling process [here.](https://arxiv.org/abs/1705.02315) 


### Dataset Contents: 

1. 112,120 frontal-view chest X-ray PNG images in 1024*1024 resolution (under images folder)
2. Meta data for all images (Data_Entry_2017.csv): Image Index, Finding Labels, Follow-up #,
Patient ID, Patient Age, Patient Gender, View Position, Original Image Size and Original Image
Pixel Spacing.


## Project Steps

### 1. Exploratory Data Analysis

The first part of this project will involve exploratory data analysis (EDA) to understand and describe the content and nature of the data. Using the output of this EDA along with contextual information about the dataset, general information about x-rays, and the clinical diagnosis of pneumonia you will curate the appropriate training and testing sets for building a pneumonia detection model. 

Note that much of the work performed during your EDA will enable the completion of the final component of this project which is focused on documentation of your algorithm for the FDA. This is described in a later section, but some important things to focus on during your EDA may be: 

* The patient demographic data (as it is available)
* The x-ray views taken (i.e. view position)
* The number of cases including: 
    * number of diseased cases,
    * number of normal cases
* The distribution of other diseases that are comorbid with pneumonia
* Number of cases per patient 
    * Also assessing development of pneumonia over time
* Pixel-level assessments of the imaging data for healthy & disease states of interest (e.g. histograms of intensity values)

### 2. Building and Training Your Model

**Training and Testing Datasets**

From your findings in the EDA component of this project, curate the appropriate training and testing sets for classifying pneumonia. Be sure to take the following into consideration: 

* Distribution of diseases other than pneumonia that are present in both datasets
* Patient make-up of training and testing sets, including demographic information, image view positions, and number of images per patient in each set
* Distribution of pneumonia-positive and pneumonia-negative cases in each dataset

**Model Architecture**

In this project, you will fine-tune an existing CNN architecture to classify x-rays images for the presence of pneumonia. There is no required archictecture required for this project, but a reasonable choice would be using the VGG16 architecture with weights trained on the ImageNet dataset. Fine-tuning can be performed by freezing your chosen pre-built network and adding several new layers to the end to train, or by doing this in combination with selectively freezing and training some layers of the pre-trained network. 


**Image Pre-Processing and Augmentation** 

You may choose or need to do some amount of  preprocessing prior to feeding imagees into your network for training and testing. This may  serve the purpose of conforming to your model's architecture and/or for the purposes of augmenting your training dataset for increasing your model performance. When performing image augmentation, be sure to think about augmentation parameters that reflect real-world differences that may be seen in chest X-rays. 

**Training** 

In training your model, there are many parameters that can be tweaked to improve performance including: 
* Image augmentation parameters
* Training batch size
* Training learning rate 
* Inclusion and parameters of specific layers in your model 

You will be asked to provide descriptions of the methods by which given parameters were chosen in the final FDA documentation.

 **Performance Assessment**

 As you train your model, you will monitor its performance over subsequence training epochs. Choose the appropriate metrics upon which to monitor test set performance. Note that 'accuracy' may not be the most appropriate statistic in this case, depending on the balance or imbalance of your validation dataset, and also depending on the clinical context that you want to use this model in (i.e. can you sacrafice high false positive rate for a low false negative rate?)

 __Note that detecting pneumonia is *hard* even for trained expert radiologists, so you should *not* expect to acheive sky-high performance.__ [This paper](https://arxiv.org/pdf/1711.05225.pdf) describes some human-reader-level F1 scores for detecting pneumonia, and can be used as a reference point for how well your model could perform.

### 3. Clinical Workflow Integration 

The imaging data provided to you for training your model was transformed from DICOM format into .png to help aid in the image pre-processing and model training steps of this project. In the real world, however, the pixel-level imaging data are contained inside of standard DICOM files. 

For this project, create a DICOM wrapper that takes in a standard DICOM file and outputs data in the format accepted by your model. Be sure to include several checks in your wrapper for the following: 
* Proper image acquisition type (i.e. X-ray)
* Proper image acquisition orientation (i.e. those present in your training data)
* Proper body part in acquisition


### 4. FDA Preparation

As the data scientist who built the classification model that will be submitted to the FDA, you will likely be responsible for two key aspects of the FDA submission process: writing an in-depth description of your algorithm and how it was trained, and writing an in-depth description of how your algorithm's performance was assessed. These two components of the submission process are described in the FDA's official documentation in sections 4 & 5 of [this document.](https://www.fda.gov/media/77635/download) 

For this project, you will complete the following steps that are derived from the FDA's official guidance on both the algorithm description and the algorithm performance assessment. __*Much of this portion of the project relies on what you did during your EDA, model building, and model training. Use figures and statistics from those earlier parts in completing the following documentation.*__

#### Algorithm Description
Provide a description of your model that covers the following: 

**1. General Information:**
* Target population that your device is intended for, including patient population organs of interest, diseases/conditions/abnormalities of interest, appropriate clinical intended to use the device (e.g. radiologist, family practice physician, nurse), and imaging modality to be used as input to your model
* Current clinical practice relevant to the diseases/conditions/abnormalities of interest
* Proposed clinical workflow including a descirption of: 
    * how your device is labeled for use in clinical practice,
    * when your device should be utilized within the proposed workflow
    * effects on interpretation time when specific claims are mae
* Device impact, including: 
    * the impact on the patient associated with device performance for both true positive and true negative marks, and
    * the impact on the patient associated with device performance for false positive and false negative marks, separately (e.g. an incorrect follow-up recommendation based on a false positive detection would likely result in short term surveillance imaging for the patient) 
* Device limitations, including diseases/conditions/abnormalities for which the device has been found ineffective and should not be used
* Supporting data from the scientific literature

**2. Algorithm Design and Function**

Include a flowchart identifying the processing, features, models, and classifiers utilized by your algorithm. The flowchart should include the following: 

* Any preprocessing steps performed by your algorithm on the original images
* The architecture of the classifier
* Any post-processing performed after classification to bring the output into a clinical environment

For each stage of your algorithm, briefly describe the design and function. You can provide references to published studies if your technique is similar, and your description should include a discussion of: 

* Purpose of the stage
* Processing steps
* Features (if any)
* Models and classifiers
* Training paradigm
* Development and training databases utilized

**3. Algorithm Training**

Algorithm training is procedure used to establish algorithm parameters and thresholds. This procedure includes the adjustment of filter parameters, the selection of the most discriminant features, and the adjustment of classifier weight and model parameters. 

Describe the criteria and performance metrics used to determine parameter settings and provide a summary of the incremental algorithm performance for appropriate intermediate stages of the algorithm. 

**4. Databases**

For the database of patient data used in training and testing your algorithm, provide specific information including: 

* The patient demographic data (as it is available)
* The radiologic techniques used and views taken
* The number of cases including: 
    * number of diseased cases,
    * number of normal cases
* The case distributions stratified by relevant confounders or effect modifiers, such as comorbid conditions

**5. Reference Standard**

The reference standard (also often called the "gold standard" or "ground truth" in the imaging community) for patient data indicates whether or not the disease is present and may include such attributes as the extent or location of the disease. 

The methodology used to establish the reference standard can impact reported performance. You should provide the rationale and describe the procedure for defining if a disease is prersent. Indiate if the reference standard is based on: 
* The output from another device
* An established clinical determination (e.g. biopsy, specific laboratory test)
* A follow-up clinical imaging examination
* A follow-up medical examination other than imaging
* An interpretation by reviewing clinician(s)

**6. Algorithm Performance Assessment**

While recommended for real-world FDA submissions,you will simply _describe_ how a standalone performance assessment would be conducted for your algorithm, rather than actually performing the assessment. This assessment helps the FDA and future users of your device to understand how well your device, by itself, identifies disease in the absense of any interaction with a clinician. Study endpoints should be selected to establish a meaningful and statistically relevant performance for the device. 

For this part of the project, you'll need to describe how you would create a reference dataset for external validation of your model in an ideal scenario. This means that you should describe the _most accurate_ way that you would create ground truth labels for this external set, and what you would compare your model's performance to. Describe what statistics you would use to evaluate performance and deem it on par with either human performance or another algorithm or tool that's out there in the real world. One helpful resource may be [this paper.](https://arxiv.org/pdf/1711.05225.pdf)

