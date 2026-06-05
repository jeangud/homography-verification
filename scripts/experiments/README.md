# Experiments
The experiments performed for the paper's results.

## Datasets
The dataset details are provided below:
### MNIST
- https://github.com/verivital/vnn-comp/tree/master/2020/PWL/benchmark/mnist/oval
- First 100 images (only one image has wrong prediction)


### CIFAR10
- https://github.com/verivital/vnn-comp/tree/master/2020/CNN/oval_framework/models
- Found first 100 images for which 3 models (base, deep, wide) have correct, and same prediction.


### MetaRoom
- https://github.com/HanjiangHu/metaroom_vnn_comp2023
- Dataset is a bit different, with one model per pertubations set for a given image.
- Wondered at first how to link a given model to its original image (around which perturbations are generated). See [Retrieve seed image from model #2](https://github.com/HanjiangHu/metaroom_vnn_comp2023/issues/2) for this discussion with the authors.
- No clear outcome there, we decided to thus find images for which the models have correct predictions, with the hope that the model is also somewhat robust to small perturbations around that image.
- Evaluated first the `proj_test` vanilla folders, but got only 48% of network accuracy.
- Then evaluated and selected the `val` vanilla folders, and got 83% accuracy. We hope this means the networks responses are a bit more robust and stable around these images.
- Ran with
```bash
$ python scripts/experiments/experiment_6/find_relevant_images.py
```
- The dataset is drawn only from those images, and we sampled 100 of those for bound computation.


### GTSRB
- https://github.com/apostovan21/vnncomp2023
- We drew 100 samples where all models have correct predictions. This helps switch easily between models while keeping the dataset fixed.
- Generated with
```shell
python generate_properties.py 0 --n 100 --epsilon 1.0 --network /home/user/work/vnn/vnncomp2023/onnx/3_48_48_
QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx
Generating 100 random specs using seed 0. Model: /home/user/work/vnn/vnncomp2023/onnx/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onn
```

### LARD
- https://github.com/deel-ai/LARD/tree/LARD_V1
- See the notebooks
