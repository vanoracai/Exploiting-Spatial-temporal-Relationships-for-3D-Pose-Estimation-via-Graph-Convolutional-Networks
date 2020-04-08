# Exploiting Spatial-temporal Relationships for 3D Pose Estimation via Graph Convolutional Networks

This is the code for the paper ICCV 2019 [Exploiting Spatial-temporal Relationships for 3D Pose Estimation via Graph Convolutional Networks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cai_Exploiting_Spatial-Temporal_Relationships_for_3D_Pose_Estimation_via_Graph_Convolutional_ICCV_2019_paper.pdf) in Pytorch.

### Dependencies

* cuda 9.0
* Python 3.6
* [Pytorch](https://github.com/pytorch/pytorch) 0.4.1.

### Dataset setup
CPN 2D detections for Human3.6 M datasets are provided by [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) by Pavllo etal., which can be downloaded by:
```bash
cd data
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
cd ..
```
3D labels and ground truth can be downloaded and put in data/ folder [3d gt labels](https://drive.google.com/file/d/1P7W3ldx2lxaYJJYcf3RG4Y9PsD4EJ6b0/view?usp=sharing)

### Download pretrained model
Pretrained models can be found in [pretrained models](https://drive.google.com/drive/folders/1cpNRX4cjd13qbb-G2XHnqnIuYOi0FNcv?usp=sharing), pls download it and put in the ckpt/ dictory(create it if it does not exist)
### Test the Model
To test on Human3.6M on single frame, run:
```bash
python main_graph.py --pad 0 --show_protocol2 --post_refine --stgcn_reload 1 --post_refine_reload 1 --previous_dir '/ckpt/1_frame/cpn/' --stgcn_model 'model_st_gcn_36_eva_post_5062.pth' --post_refine_model 'model_post_refine_36_eva_post_5062.pth' 
```

To test on Human3.6M on 3-frames, run:
```bash
python main_graph.py --pad 1 --show_protocol2 --post_refine --stgcn_reload 1 --post_refine_reload 1 --previous_dir '/ckpt/3_frame/cpn/' --stgcn_model 'model_st_gcn_58_eva_post_4903.pth' --post_refine_model 'model_post_refine_58_eva_post_4903.pth' 
```
### Train the Model
To train on Human3.6M with 3-frame, run:
```bash
python main_graph.py --pad 1 --pro_train 1 --save_model 1
```
After training for several epoches, add post_refine part
```bash
python main_graph.py --pad 1 --pro_train 1 --post_refine --save_model 1 --learning_rate 1e-5 --sym_penalty 1 --co_diff 1  --stgcn_reload 1  --previous_dir [your model saved path] --stgcn_model [your pretrained model] 
```

### Citation
```bash
@inproceedings{cai2019exploiting,
  title={Exploiting spatial-temporal relationships for 3d pose estimation via graph convolutional networks},
  author={Cai, Yujun and Ge, Liuhao and Liu, Jun and Cai, Jianfei and Cham, Tat-Jen and Yuan, Junsong and Thalmann, Nadia Magnenat},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2272--2281},
  year={2019}
}
```
### Acknowledgements
Some of our implementation code/preprocessed data was adapted from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) by Pavllo et al., [st-gcn](https://github.com/yysijie/st-gcn) by Yansijie et al., [simple-yet-effective baseline](https://github.com/una-dinosauria/3d-pose-baseline) by Julia et al.,[Non-local neural networks](https://arxiv.org/abs/1711.07971). Thanks for their help!

### Licence
MIT
