## Requirement
  ### Hardware
  - CUDA-ready GPU with Compute Capability 7.0+
  - 24 GB VRAM (to train to paper evaluation quality)
  ### Software
  - Conda
  - C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
  - CUDA SDK 11 for PyTorch extensions (比如11.8，不能用11.6和12+)
  - C++ Compiler and CUDA SDK must be compatible
## 环境
```shell
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
conda env create --file environment.yml
conda activate gaussian_splatting
```
## 视频转图片
```shell
# 指定视频路径和fps
ffmpeg -i E:\\Projects\\3d_gaussian\\datasets\\cone\\VID_20240903_133929.mp4 -qscale:v 1 -qmin 1 -vf fps=3 %04d.jpg
```
生成的图片全部放在名为input的文件夹中
## Sfm
```shell
# 输入的文件夹里要包含input文件夹
python convert.py -s E:\\Projects\\3d_gaussian\\datasets\\cone
```
comap bin文件转txt
```shell
colmap model_converter --input_path E:\Projects\3d_gaussian\datasets\cone\sparse\0 --output_path E:\Projects\3d_gaussian\datasets\cone\sparse\0 --output_type TXT
```
## Train
```shell
# 输入的文件夹里要包含之前生成的文件夹，sparse, stereo....
python train.py -s ../datasets/cone
```
## 常用可选参数：
--resolution / -r

  Specifies resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.

--iterations

  Number of total iterations to train for, 30_000 by default.
--save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, 7000 30000 <iterations> by default.

--checkpoint_iterations

  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.

--start_checkpoint

  Path to a saved checkpoint to continue training from.
7. Output
在output文件夹里
8. 可视化
见readme Interactive Viewers部分
9. render想要的view
