# Effect of Pruning on ResNet Architectures

**Authors:** Anchal Agrawal (aa11597), Ajay Sudharshan Satish (as18476)

## Description 
The project focuses on studying the effects of pruning techniques on ResNet architectures, aiming to analyze their impact on model performance, computational efficiency, and memory usage. Additionally, the project explores the favorable conditions and scenarios where pruning yields optimal results, balancing trade-offs between accuracy and efficiency. It is often observed that pruning a trained model results in a drop in accuracy, especially when the pruning process is not carefully optimized. This project aims to investigate strategies to minimize this accuracy drop in ResNet architectures, while maintaining or enhancing computational efficiency.

Keeping this in mind, we have implemented two pruning techniques:L1 Norm Pruning and Unstructured Weight Level Pruning. The objective is to reduce model size, computational complexity, and memory usage while maintaining performance. L1 Norm Pruning reduces parameters and FLOPs in a structured manner, whereas Unstructured Weight Level Pruning uses a mask-based approach to prune individual weights, significantly improving computational efficiency and throughput. Results show substantial reductions in CPU and CUDA time, latency, and memory usage, making these models more suitable for resource-constrained environments. 

In addition, the project also explores the use of Depthwise Separable Convolutions for reducing computational cost. However, contrary to expectations, applying pruning techniques to models with depthwise separable convolutions resulted in opposite effects—performance degradation rather than improvements. This was an unexpected outcome, suggesting that pruning depthwise convolutions may interfere with the inherent design of the convolutional layers, which rely on a more compact structure to function efficiently.


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
  - Configured the Resnet architecture with Deptwise Convolution and Squeeze-and-Excitation Block
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
- **Pruning with Deptwise-Separable Convolution architecture**
    ```
    Resnet50_DE.ipynb
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
  3) Finetuning Pruned Resnet: `finetune.py`
     ```
      python finetune.py --depth 50 --dataset cifar100
      ```


- **Soft Filter Norm pruning**
  1) Baseline ResNet50 model:  `main.py`
     ```
      python main.py --depth 50 --dataset cifar100
      ```
  2) Pruning ResNet50 model: `res50prune.py`
     ```
      python res50prune.py --depth 50 --dataset cifar100
      ```

  4) Finetuning Pruned Resnet: `finetune.py`
     ```
      python finetune.py --depth 50 --dataset cifar100
      ```

## Results

- **L1 Norm Pruning:**
  Reduced model complexity, with a decrease in parameters and FLOPs, leading to improved throughput and reduced latency.
- **Unstructured Weight Level Pruning:**
  Showed significant reductions in computational time, including CPU and CUDA time, along with improved latency per batch and higher throughput.
- **Depthwise Separable Convolutions:**
  Contrary to expectations, applying pruning techniques to models using depthwise separable convolutions resulted in performance degradation rather than improvement. This suggests that pruning may interfere with the efficiency of depthwise convolutions, which are already optimized for reducing computation.

## Conclusion

Both L1 Norm and Unstructured Weight Level Pruning effectively reduced model complexity, improving computational efficiency, throughput, and latency. However, the use of Depthwise Separable Convolutions yielded unexpected results, as pruning these layers led to performance degradation rather than the anticipated improvements. This suggests that depthwise convolutions, already optimized for computational efficiency, may be sensitive to pruning, requiring a more nuanced approach to optimization. Overall, while pruning can be highly effective in reducing model size and enhancing performance, its impact varies depending on the architecture, and further exploration is needed to understand how pruning interacts with different model components.

### Base ResNet26 model
Output 
![Alt text](https://github.com/Ajsat3801/HPML-Project/blob/main/Documentation/ResNet26_base_output.png "Architecture")
