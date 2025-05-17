# HPML Project: Analyzing Intermediate Steps of NVCC Compilation of CUDA Kernels

### ****Disclaimer****: This repository is under constant development. If you have questions, feel free to raise an Issue and use the "question" label. For suggestions, use "enhancement". Issues that are not labeled as "question" or "enhancement" are assumed to indicate an error.

## Author
IncipioVincero

---

## 1. Objectives
<!--Describe the task being solved/researched-->
- Review the compilation steps carried out by NVIDIA's NVCC compiler [see documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- Examine ptxas compilation steps to decipher connection between PTX and SASS
---

## 2. Static Analysis of PTX/SASS
<!--Summarize the model architecture(s) used (e.g., ResNet-18, Transformer). Include:
- Framework (e.g., PyTorch, TensorFlow)
- Any custom layers or changes to standard models-->
See the [cuda.commands](/src/Utilities/cuda.commands) file to view sample commands that can be used for analysis. 


---


### A. Requirements

<!--Install dependencies:
```bash
pip install -r requirements.txt
```-->
- **H100 GPU** (sm_90/90a, compute_90)
- Linux based operating system 

---

B. Wandb Dashboard

[Companion wandb dashboard](https://wandb.ai/kw_columbia?shareProfileType=copy)

---

<!--To train the model from scratch:
```bash
python train.py --config configs/default.yaml
```-->


---

### E. Quickstart: Minimum Reproducible Result

**Compilation**

**NVCC Dryrun**
- Modify the variables in [finalTemplate.sh](/src/Utilities/finalTemplate.sh) script to suit your use case and run : ./finalTemplate.sh

   | Variables | Description |
   | --------- | ------------|
   |  ARCH         | GPU Architecture (e.g. sm_90, etc.)            |
   |  FILES          | List of files to compile            |
   | SOURCE_PATH      | Path to directory containing target files |
<!--To reproduce our minimum reported result (e.g., XX.XX% accuracy), run:

```bash
# Step 1: Set up environment
pip install -r requirements.txt

# Step 2: Download dataset
bash scripts/download_dataset.sh  # if applicable

# Step 3: Run training (or skip if checkpoint is provided)
python train.py --config configs/default.yaml

# Step 4: Evaluate
python eval.py --weights checkpoints/best_model.pth
```-->

---

## 5. Notes 
TBD
<!-- - All scripts are located in `scripts/`, `train.py`, `eval.py`, and `configs/`.
- Trained Model are saved in `models/`.
- Contact information-->
