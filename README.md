# Fish Weight Estimation from Image

Non-invasive weight prediction of fish using image segmentation and regression.

## Method
1. Interactive segmentation with **Segment Anything Model (SAM ViT-H)**
2. Area calculation in cm² (pixel-to-cm calibration)
3. Length estimation via fitted ellipse
4. Weight estimation using power-law regression:  
   **W = a × Areaᵇ**  
   (a = 0.01, b = 1.5 – values from [Deep Learning image segmentation for extraction of fish body measurements and prediction of body weight and carcass traits in Nile tilapia
, 2020])

Reference:  
Deep Learning image segmentation for extraction of fish body measurements and prediction of body weight and carcass traits in Nile tilapia
Arthur Francisco Araujo Fernandes
Eduardo Maldonado Turra
Érika Alvarenga
Tiago Luciano Passafaro

(Formula adapted from the paper)

## Features
- Point-prompt interactive segmentation
- Automatic area & length computation
- CSV export of results
- Output images with mask overlay

## Output example
<img width="879" height="394" alt="output_fish_1" src="https://github.com/user-attachments/assets/13a90be5-a5a1-4999-9f85-79dda14fd9a7" />
<img width="800" height="557" alt="segmented_fish_5" src="https://github.com/user-attachments/assets/2b7bb5a2-6160-4e08-a226-542a8f56c50e" />



## Requirements
```txt
torch torchvision
segment-anything
opencv-python
numpy matplotlib
