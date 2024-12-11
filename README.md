# Effect of Pruning on ResNet Architectures

**Authors:** Anchal Agrawal (aa11597), Ajay Sudharshan Satish (as18476)

The project focuses on studying the effects of pruning techniques on ResNet architectures, aiming to analyze their impact on model performance, computational efficiency, and memory usage. Additionally, the project explores the favorable conditions and scenarios where pruning yields optimal results, balancing trade-offs between accuracy and efficiency. It is often observed that pruning a trained model results in a drop in accuracy, especially when the pruning process is not carefully optimized. This project aims to investigate strategies to minimize this accuracy drop in ResNet architectures, while maintaining or enhancing computational efficiency.

Keeping this in mind, we have implemented two types of pruning techniques: structured pruning and unstructured pruning. Structured pruning involves removing entire filters or channels, simplifying the model's architecture and making it more hardware-friendly. On the other hand, unstructured pruning removes individual weights, offering finer granularity but requiring additional considerations for efficient deployment. Both approaches are analyzed to evaluate their impact on model accuracy, computational efficiency, and overall performance.


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
Each model has a dedicated notebook: 
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

## Results

### Base ResNet26 model
Output 
![Alt text](https://github.com/Ajsat3801/HPML-Project/blob/main/Documentation/ResNet26_base_output.png "Architecture")
