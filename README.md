# Segmentation Using ResNet26 with Depth-Wise Separable Convolution and Squeeze-and-Excitation Blocks

**Authors:** Anchal Agrawal (aa11597), Ajay Sudharshan Satish (as18476)

The project focuses on developing efficient segmentation models using the ResNet26 architecture enhanced with depth-wise separable convolutions and Squeeze-and-Excitation (SE) blocks. The aim is to balance computational efficiency and segmentation accuracy while reducing model size. Four models are implemented: a baseline ResNet26, a ResNet26 with depth-wise separable convolutions, a pruned version of the depth-wise separable model, and a combined model integrating both depth-wise separable convolutions and SE blocks. The models are trained and evaluated on Cityscapes using advanced optimization techniques to ensure practical applicability in resource-constrained environments.

## Directory Structure

```
├── Documentation           // Contains images and results used for documentation  
├── Models                  // Contains the trained models  
├── Notebooks               // Contains notebooks for training and benchmarking the models  
```

## Running the project

### Installing Dependencies
```
pip install -r requirements.txt
```

### Running the models
Each model has a dedicated notebook: 
1) Basic ResNet26 model:  `Notebooks/ResNet26_base.ipynb`
2) ResNet26 with Depthwise separable convolutions: `Notebooks/ResNet26_DS.ipynb`
3) ResNet26 with Depthwise separable convolutions and SE blocks: `Notebooks/ResNet26_DS_SE.ipynb`
4) Pruned ResNet26 with Depthwise separable convolutions: `Notebooks/ResNet26_DS_Pruned.ipynb`

## Results

### Base ResNet26 model
Output 
![Alt text](https://github.com/Ajsat3801/HPML-Project/blob/main/Documentation/ResNet26_base_output.png "Architecture")
