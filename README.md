![plantcamo](https://github.com/yjybuaa/PlantCamo/assets/39208339/9b6888db-cd9d-46f0-b851-d40726788cf4)

<div align=center>
<a src="https://img.shields.io/badge/%F0%9F%93%96-Arxiv_2410.17598-red.svg?style=flat-square" href="https://arxiv.org/abs/2410.17598">
<img src="https://img.shields.io/badge/%F0%9F%93%96-Arxiv_2410.17598-red.svg?style=flat-square">
</a>
<a src="https://img.shields.io/badge/%F0%9F%9A%80-PyTorch_Version-ed6c00.svg?style=flat-square" href="https://github.com/yjybuaa/PlantCamo">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-PyTorch_Version-ed6c00.svg?style=flat-square">
</a>
</div>

__PlantCamo Dataset__ is the first dataset dedicated for plant camouflage detection. It contains over 1,000 images with plant camouflage characteristics.


## Train

Download `pvt_v2_b2.pth` at [here](https://pan.baidu.com/s/11dSkyGKb71lT_7HxSCiIjw) (Code: gy87) or [Google drive link](https://drive.google.com/file/d/1-AFs2dP3p0OMw3Vbnf0tMsgkYbpT6C23/view?usp=sharing), and put it into `.\pretrained_pvt`

Download `PlantCamo-Train-and-Test` at [here](https://pan.baidu.com/s/1vdR-kj63qvsT3M4-wkgMoQ)(Code: hq87) or [Google drive link](https://drive.google.com/file/d/1eMvSbNJJbh6BYea-3ZktzDReHFujsb1_/view?usp=drive_link), and put it into `.\datasets`

### Kaggle

To start with, you should copy the notebook  [CUDA 11.8](https://www.kaggle.com/code/seachenbgdy/1-cuda11-8-torch2-0-0)

Then, you only run the following cells

```bibtex
!pip install jittor
!pip install pysodmetrics
!pip install "numpy<1.23.0" "scipy>=1.8.0" "scikit-image>=0.19.0" --force-reinstall
```
```bibtex
!sudo apt-get update
!sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
!mpirun --allow-run-as-root -np 2 python Train.py
```

## Evaluation

Firstly, you should run the `MyTest.py`

Secondly, you should run the `metric_caller.py` and you can find it in [PySODMetrics](https://github.com/lartpang/PySODMetrics)

|  | $S_\alpha \uparrow$ | $F_\beta^\omega \uparrow$ | $M \downarrow$ | $E_\varphi^{ad} \uparrow$ | $E_\varphi^m \uparrow$ | $E_\varphi^{max} \uparrow$ | $F_\beta^{ad} \uparrow$ | $F_\beta^m \uparrow$ | $F_\beta^{max} \uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pytorch** | 0.869 | 0.812 | 0.030 | 0.932 | 0.929 | 0.933 | 0.843 | 0.845 | 0.854 |
| **Jittor** | 0.875 | 0.813 | 0.030 | 0.932 | 0.934 | 0.939 | 0.838 | 0.845 | 0.858 |


## Citation
```bibtex
@article{plantcamo,
      title={PlantCamo: Plant Camouflage Detection}, 
      author={Jinyu Yang and Qingwei Wang and Feng Zheng and Peng Chen and Aleš Leonardis and Deng-Ping Fan},
      journal={CAAI Artificial Intelligence Research (AIR)},
      year={2025}
}
```
