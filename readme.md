## 红细胞图片目标检测

### 实验背景
该实验是基于四百张红细胞显微图，使用Transfer Learning(迁移学习)和Faster RCNN，在Azure数据科学虚拟机里利用预装的Keras和Tensorflow框架完成红细胞的检测。

### 实验环境
- Azure数据科学虚拟机：NC6类型，Ubuntu
- Keras 2.0.9
- Tensorflow 1.4.0
- Anaconda Python 3.5.2 with CUDA 8.0

#### 1. 虚拟机准备
1) 创建Azure 数据科学虚拟机
在全球版Azure的管理门户上搜索Azure数据科学虚拟机(Data Science Virtual Machine)，开始创建。注意选择Ubuntu系统，HDD磁盘类型，NC6型号虚拟机。创建成功后，通过Putty连接虚拟机。

2) 训练环境配置
配置Keras默认后台为Tensorflow。也可以在/home/<username>/.keras里面找到keras.json文件。
如果使用Python环境运行代码，可以运行source activate py35，启动虚拟环境，也可以直接使用Python。

### 数据说明
数据里包含约四百张红细胞显微图片，以及对应的Pascal VOC格式的标注(xml)，[下载]((https://github.com/cosmicad/dataset))

<img src="image/BloodImage.jpg" width="400" height="280" />

### 参考来源
-  https://github.com/yhenon/keras-frcnn/tree/38fe0d77a11293e9cac43fe889d08c1fe23713d6

### 代码说明

- pascal_voc_parser.py: 数据预处理，读取图片和其VOC格式(xml)的标注，转换成"path/image.jpg,x1,y1,x2,y2,class_name"这种格式。
   - 输入为数据路径，图片存储在JPEGImages中，标注存储在Annotations中。
   - 目前文件中将路径中的所有图片都设定为训练集，如果分割测试集，需要调整代码。
- train_frcnn.py: 模型训练。
   - get_data函数需要设置数据存储的路径(即dataset的路径)
   - 通过epoch_length和num_epochs修改训练次数和每一次的训练长度。 
   - 代码里面没有设置validation set，只针对训练集。
- test_frcnn.py: 测试模型。
   - 将图片放到测试文件夹中。
   - 需要读取训练时保存的config.pickle文件和训练好的模型model.hdf5。

