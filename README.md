# NYCU Computer Vision 2025 Spring HW4

**Author:** 黃皓君  
**Student ID:** 111550034  

---

## Introduction

This work implements **multi-task blind image restoration** removing rain streaks and snow using the **PromptIR** framework. PromptIR injects learnable prompt vectors into a shared transformer-based encoder–decoder so one network can adapt to four different degradations simply by switching prompts.

---
## Installation and Data Preparation

The Conda environment used can be recreated using the env.yml file
```
conda env create -f env.yml
```

The training data should be placed in ``` data/Train/{task_name}``` directory 

```
└───Train
    ├───Desnow
    │   ├───gt	    (snow_clean-*.png)
    │   └───snowy	(snow_clean-*.png)
    └───Derain
        ├───gt	    (rain_clean-*.png)
        └───rainy	(rain-*.png)
```

The testing data should be placed in the ```data/test/degraded```

---
## How to Run  

### 1. Train
```bash
python train.py
```
### 2. Test
```bash
python test.py --ckpt_name --output_path
```
### 3. Convert outputs to NPZ
```bash
python img2npz.py --folder_path
```
---
## Results
Experimental Results 
| Loss configuration | PSNR public/private | 
| --- | --- | 
| L₁ only | 28.40 / 27.66 | 
| MS-SSIM only | 30.85 / 30.22 | 
| L₁ + MS-SSIM | 31.03 / 30.31 | 
| **L₁ + MS-SSIM + grad + charb** | **31.04 / 30.53** |

---
## Performance Snapshot  
![shapshot](./shapshot.png) 
