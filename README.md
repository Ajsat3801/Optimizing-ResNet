# Effect of Pruning on ResNet Architectures

**Authors:** Anchal Agrawal (aa11597), Ajay Sudharshan Satish (as18476)

The project focuses on studying the effects of pruning techniques on ResNet architectures, aiming to analyze their impact on model performance, computational efficiency, and memory usage. Additionally, the project explores the favorable conditions and scenarios where pruning yields optimal results, balancing trade-offs between accuracy and efficiency. It is often observed that pruning a trained model results in a drop in accuracy, especially when the pruning process is not carefully optimized. This project aims to investigate strategies to minimize this accuracy drop in ResNet architectures, while maintaining or enhancing computational efficiency.

Keeping this in mind, we have implemented two pruning techniques: L1-norm pruning and Soft Filter Norm pruning. L1-norm pruning removes filters based on their L1 norm, assuming that filters with smaller L1 norms contribute less to the model's output and can be pruned with minimal impact on accuracy. Soft Filter Norm pruning, on the other hand, applies a soft threshold to filter norms, enabling more flexibility and adaptability in the pruning process. These techniques are analyzed in the context of ResNet architectures to study their effects on model accuracy, computational efficiency, and performance trade-offs.


## Directory Structure

```
├── Documentation           // Contains images and results used for documentation  
├── Models                  // Contains the trained models  
├── Notebooks               // Contains notebooks for training and benchmarking the models  
```
## Project Milestones

### Milestone 1: Setup and Baseline Measurement
- **Status:** Completed  
- **Activities:**
  - Configured the ResNet model.
  - Measured baseline performance metrics, including accuracy, inference time, and model size.

### Milestone 2: Implement L1-Norm Pruning
- **Status:** Completed  
- **Activities:**
  - Applied L1-norm pruning to remove filters based on their L1 norm.
  - Tested the impact on accuracy and computational efficiency at different pruning levels (e.g., 30%, 60%, 90%).

### Milestone 3: Implement Soft Filter Norm Pruning
- **Status:** Completed  
- **Activities:**
  - Applied Soft Filter Norm pruning with a dynamic threshold to remove less important filters.
  - Analyzed its effects on accuracy and efficiency compared to L1-norm pruning.

### Milestone 4: Evaluation and Analysis
- **Status:** Completed  
- **Activities:**
  - Compared the two pruning methods in terms of:
    - Model accuracy
    - Inference time
    - Model size
    - Trade-offs between accuracy and computational efficiency.

## Running the project

### Installing Dependencies
```
pip install -r requirements.txt
```

### Running the models
- **Soft Filter Norm pruning**
  1) Baseline ResNet50 model:  `main.py`
     ```
      python main.py --depth 50 --dataset cifar100
      ```
  2) Pruning ResNet50 model: `res50prune.py`
     ```
      python res50prune.py --depth 50 --dataset cifar100
      ```
  3) Finetuning Pruned Resnet: `finetune.py`
     ```
      python finetune.py --depth 50 --dataset cifar100
      ```
  4) Finetuning Pruned Resnet with fixed epoch: `finetune_E.py`
     ```
      python finetune_E.py --depth 50 --dataset cifar100
      ```
  5) Finetuning Pruned Resnet with fixed Computaional budget: `finetune_B.py`
     ```
      python finetune_B.py --depth 50 --dataset cifar100
      ```

  - **L1 Norm Pruning**
  1) Baseline ResNet50 model:  `main.py`
     ```
      python main.py --depth 50 --dataset cifar100
      ```
  2) Pruning ResNet50 model: `res50prune.py`
     ```
      python res50prune.py --depth 50 --dataset cifar100
      ```
  3) Finetuning Pruned Resnet with fixed epoch: `finetune_E.py`
     ```
      python finetune_E.py --depth 50 --dataset cifar100
      ```
  4) Finetuning Pruned Resnet with fixed Computaional budget: `finetune_B.py`
     ```
      python finetune_B.py --depth 50 --dataset cifar100
      ```

## Results

### Base ResNet26 model
Output 
![Alt text](https://github.com/Ajsat3801/HPML-Project/blob/main/Documentation/ResNet26_base_output.png "Architecture")
