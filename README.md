# VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition

　An on going TF implementation on VoxNet to deal with 3D LiDAR pointcloud segmentation classification, refer to [paper](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf).

```bibtex
@inproceedings{Maturana2015VoxNet,
  title={VoxNet: A 3D Convolutional Neural Network for real-time object recognition},
  author={Maturana, Daniel and Scherer, Sebastian},
  booktitle={Ieee/rsj International Conference on Intelligent Robots and Systems},
  pages={922-928},
  year={2015},
}
```

<p align="center">
   <img src="./readme/The VoxNet Architecture.png" width="420" alt="" />
</p>
Architecture Details 

* **Input Layer**: Accepts a fixed-size grid of `32×32×32` voxels. Values are updated depending on the occupancy model (Hit Grid, Binary, or Density), resulting in a (-1, 1) range.
* **Convolutional Layers** `Conv(f, d, s)`:
* Learns `f` feature maps by convolving input with filters of shape `d×d×d`.
* Uses **Leaky ReLU** activation (0.1).


* **Pooling Layers** `Pool(m)`:
* Downsamples input by factor `m` using Max Pooling on non-overlapping blocks.


* **Fully Connected Layer** `FC(n)`:
* Learns linear combinations of features.


* **Output Layer**:
* Uses **Softmax** nonlinearity to provide probabilistic output for `K` classes.


* **VoxNet Structure**: `Conv(32, 5, 2)` → `Conv(32, 3, 1)` → `Pool(2)` → `FC(128)` → `FC(K)`

## Dataset

* 
[Sydney Urban Object Dataset (SUOD)](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml) 



## Requirements

Implemented and tested on **macOS (CPU)** and **Linux** with **Python 3.5** and **TensorFlow 1.3.0**.

1. **Clone this repo:**
```bash
$ git clone https://github.com/Durant35/VoxNet --recurse-submodules
$ cd VoxNet

```


2. **Setup virtual environment:**
*Using Conda (Recommended):*
```bash
$ conda create -n voxnet_env python=3.5
$ conda activate voxnet_env
$ pip install -r requirements.txt

```


*Note: `python-pcl` is **not** required for this version.*

## 1. Data Pre-process

Generate voxelized training/testing data (`.npy` files) from the SUOD dataset. This step converts point clouds into the 3D grids required by the network.

**Note:** We use the `--nopcd` flag to skip generating legacy `.pcd` files, and we removed the dependency on the external PCL library.

```bash
# Make sure you are in the root directory
$ python ./src/preprocess.py -h

# Clear previous cache (if any) and generate data for Folds 1 & 2
$ python ./src/preprocess.py --clear_cache
$ python ./src/preprocess.py --fold 1 --nopcd
$ python ./src/preprocess.py --fold 2 --nopcd

# Generate testing data (Fold 3)
$ python ./src/preprocess.py --fold 3 --type testing --nopcd

```

## 2. Training

Train the VoxNet model on the generated `.npy` data.

**Warning for CPU Users (MacBook/Laptop):** Training on a CPU is significantly slower than on a GPU. For a quick test, reduce the number of epochs.

```bash
# Run training (clearing old logs to start fresh)
$ python ./src/train.py --clear_log --batch_size 16 --num_epochs 8

```

*Expected Output:*

```text
Start training...
INFO:tensorflow:loss = 0.6589, step = 10 (2.5 sec)
...
INFO:tensorflow:Saving checkpoints for ... into ./logs/model.ckpt.
Finished training.

```

## 3. Testing / Evaluation

Evaluate the trained model on the test set (Fold 3).

```bash
# Run evaluation script
$ python ./src/eval.py

```

*Expected Output:*

```text
Predicted: trunk, Ground Truth: trunk
Predicted: pedestrian, Ground Truth: pedestrian
...
INFO:tensorflow:Restoring parameters from ./logs/model.ckpt-1380
INFO:tensorflow:Finished evaluation
INFO:tensorflow:Saving dict for global step 1380: accuracy = 0.6573034, loss = 3.78
Finished testing.

```

## Visualization

You can visualize the training progress (Loss, Accuracy) using TensorBoard:

```bash
$ tensorboard --logdir=./logs

```

Then open `http://localhost:6006` in your browser.
