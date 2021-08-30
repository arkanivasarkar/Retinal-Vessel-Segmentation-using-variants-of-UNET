# Retinal-Vessel-Segmentation-using-Variants-of-UNET

This repository contains the implementation of fully convolutional neural networks for segmenting retinal vasculature from fundus images.

![alt text](https://i.ibb.co/BKr7cVF/Picture1.png) 


Four architecures/models were made keeping U-NET architecture as the base.
The models used are:
- Simple U-NET
- Residual U-NET (Res-UNET)
- Attention U-NET
- Residual Attention U-NET (RA-UNET)

The performance metrics used for evaluation are accuracy and mean IoU.


## Methods
Images from HRF, DRIVE and STARE datasets are used for training and testing. The following pre-processing steps are applied before training the models:
- Green channel selection
- Contrast-limited adaptive histogram equalization (CLAHE)
- Cropping into non-overlapping patches of size 512 x 512

10 images from DRIVE and STARE and 12 images from HRF was kept for testing the models. The training dataset was then split into 70:30 ratio for training and validation.

Adam optimizer with a learning rate of 0.001 was used as optimizer and IoU loss was used as the loss function. The models were trained for 150 epochs with a batch size of 16, using NVIDIA Tesla P100-PCIE GPU. 

## Results
The performance of the models were evaluated using the test dataset.
Out of all the models, Attention U-NET achieved a greater segmentation performance. 


The following table compares the performance of various models

| **Datasets** |    **Models**    | **Average Accuracy**| **Mean IoU**|
|:------------:|:----------------:|:-------------------:|:-----------:|
| HRF          | Simple U-NET     | 0.965               |0.854        |
| HRF          | Res-UNET         | 0.964               |0.854        |
| HRF          | Attention U-NET  | 0.966               |0.857        |
| HRF          | RA-UNET          | 0.963               |0.85         |
| DRIVE        | Simple U-NET     | 0.9                 |0.736        |
| DRIVE        | Res-UNET         | 0.903               |0.741        |
| DRIVE        | Attention U-NET  | 0.905               |0.745        |
| DRIVE        | RA-UNET          | 0.9                 |0.735        |
| STARE        | Simple U-NET     | 0.882               |0.719        |
| STARE        | Res-UNET         | 0.893               |0.737        |
| STARE        | Attention U-NET  | 0.893               |0.738        |
| STARE        | RA-UNET          | 0.891               |0.733        |

![alt text](https://i.ibb.co/W07sGYv/Picture3.png)



### Datasets
The datasets of the fundus images can be acquired from:
1. [HRF](https://www5.cs.fau.de/research/data/fundus-images/)
2. [DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/)
3. [STARE](https://cecas.clemson.edu/~ahoover/stare/)

The trained models are present in `Trained models` folder.



## References

[1] Vengalil, Sunil Kumar & Sinha, Neelam & Kruthiventi, Srinivas & Babu, R. (2016). Customizing CNNs for blood vessel segmentation from fundus images. 1-4. 10.1109/SPCOM.2016.7746702..

[2] Ronneberger O., Fischer P., Brox T. U-Net: Convolutional networks for biomedical image segmentation International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer (2015), pp. 234-241

[3] Zhang, Zhengxin & Liu, Qingjie. (2017). Road Extraction by Deep Residual U-Net. IEEE Geoscience and Remote Sensing Letters. PP. 10.1109/LGRS.2018.2802944.

[4] Oktay, Ozan & Schlemper, Jo & Folgoc, Loic & Lee, Matthew & Heinrich, Mattias & Misawa, Kazunari & Mori, Kensaku & McDonagh, Steven & Hammerla, Nils & Kainz, Bernhard & Glocker, Ben & Rueckert, Daniel. (2018). Attention U-Net: Learning Where to Look for the Pancreas.

[5] Ni, Zhen-Liang & Bian, Gui-Bin & Zhou, Xiao-Hu & Hou, Zeng-Guang & Xie, Xiao-Liang & Wang, Chen & Zhou, Yan-Jie & Li, Rui-Qi & Li, Zhen. (2019). RAUNet: Residual Attention U-Net for Semantic Segmentation of Cataract Surgical Instruments.

[6] Jin, Qiangguo & Meng, Zhaopeng & Pham, Tuan & Chen, Qi & Wei, Leyi & Su, Ran. (2018). DUNet: A deformable network for retinal vessel segmentation.

&nbsp;
&nbsp;
&nbsp;
&nbsp;




<h3><strong> This project is done during Indian Academy of Sciences Summer Reasearch Fellowship '21 </strong></h3>
