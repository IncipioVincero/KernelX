# HPML Project: Analyzing Intermediate Steps of NVCC Compilation of CUDA Kernels

### ****Disclaimer****: This repository is under constant development. If you have questions, feel free to raise an Issue and use the "question" label. For suggestions, use "enhancement". Issues that are not labeled as "question" or "enhancement" are assumed to indicate an error.

## Author
IncipioVincero

---

## 1. Objectives
<!--Describe the task being solved/researched-->
- Review the compilation steps carried out by NVIDIA's NVCC compiler [https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/]
- Examine ptxas compilation steps to decipher connection between PTX and SASS
---

## 2. Static Analysis of PTX/SASS and Experimental Inline PTX 
<!--Summarize the model architecture(s) used (e.g., ResNet-18, Transformer). Include:
- Framework (e.g., PyTorch, TensorFlow)
- Any custom layers or changes to standard models-->
- Two convolution kernels are presented: one with inline ptx add to manage memory and the other without it.
- Kernels are compiled to PTX and SASS using **compilation.sh** script [Path: KernelX/src/Compilation/compilation.sh]
- .SASS and .ptx files are created to allowe for comparison of instructions used.
- 

---
## 3. Model Integration
- Both convolution kernels are incorporated in small CNN model.
- **model_integration.ipynb** file shows how to setup and run CNN models with each kernel, view timing results and generate roofline model.



---
## 4. Final Results Summary



| Kernel               | Stand-alone Execution Time        | Execution Time within Model |
|----------------------|-------------|--------------------|                
| Convolution (non-inline) | XX.XX%      |                    |
| Convolution (inline ptx)   | XX.XX ms    |                    |
| Model Size           | XX MB       |                    |
| Peak Memory Use      | XX MB       |                    |
| Training Time/Epoch  | XX s        |                    |
| Device               | A100, Jetson Nano, M1 Pro, etc.  |

---

## 4. Reproducibility Instructions

### A. Requirements

<!--Install dependencies:
```bash
pip install -r requirements.txt
```-->

---

B. Wandb Dashboard

<!--View training and evaluation metrics here: Wandb Dashboard Link
(Replace with actual link)-->

---

### C. Specify for Training or For Inference or if Both 

<!--To train the model from scratch:
```bash
python train.py --config configs/default.yaml
```-->

---

### D. Evaluation

<!--To evaluate the trained model:
```bash
python eval.py --weights checkpoints/best_model.pth
```-->

---

### E. Quickstart: Minimum Reproducible Result

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

## 5. Notes (up to you)
<!-- - All scripts are located in `scripts/`, `train.py`, `eval.py`, and `configs/`.
- Trained Model are saved in `models/`.
- Contact information-->
