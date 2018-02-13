## Cosmicad detection

### Background
This experiment is based on 400 cosmicad micrographs, with Transfer Learning and Faster RCNN, in Azure Data Science Virtual Machine and its pre-installed Keras and Tensorflow framework to implement cosmicad detection.

### Experiment environment
- Azure Data Science Virtual Machine: NC6 size, and Ubuntu system
- Keras 2.0.9
- Tensorflow 1.4.0
- Anaconda Python 3.5.2 with CUDA 8.0

1) Create Azure Data Science Virtual Machine

Search Azure Data Science Virtual Machine in global Azure portal and start creating. Notice to choose Ubuntu system, HDD type, NC6 size virtual machine. After created, connect to it with Putty.

2) Training environment configuration

Config Keras's backend as Tensorflow. You can find keras.json file in "/home/<username>/.keras".
If you want to use Python environment, you can run "source activate py35" to start virtual environment or directly use Python.

### Dataset introduction
There are 400 cosmicad micrograph images in the dataset, and Pascal VOC format label (xml)，[Download]((https://github.com/cosmicad/dataset))

<img src="image/BloodImage.jpg" width="400" height="280" />

### Github reference
-  https://github.com/yhenon/keras-frcnn/tree/38fe0d77a11293e9cac43fe889d08c1fe23713d6

### Code description

- pascal_voc_parser.py: Data preprocessing. Read images and VOC format labels (xml), and transform them into format of "path/image.jpg,x1,y1,x2,y2,class_name".
   - Input is dataset path, images stored in JPEGImages folder, labels stored in Annotations folder.
   - All images in the path are set as training set.
- train_frcnn.py: Model training
   - get_data function needs to be configured with data store path (path of dataset)
   - epoch_length and num_epochs are for number of iterations and training length in each iteration. 
   - No validation set  in the code.
- test_frcnn.py: Test the model.
   - Put the images in the test folder.
   - Files of config.pickle and model.hdf5 needs to be loaded when test the data.