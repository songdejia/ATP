## Requirements 
#### Hardware Requirements
* GPU: GeForce GTX 1080 Ti
* CPU: CPU E5-2620 v4 @ 2.10GHz
 
#### Software dependencies
* OS:  Ubuntu 18.04.2 LTS
* GCC 5.3
* Matlab 2018
* Anaconda 3
* CUDA 9.0, CUDNN 7.0.5
* Python libraries: PyTorch 0.4.1, OpenCV, NumPy etc.

We have uploaded our anaconda environments containing all needed python libraries, please follow steps below for installation.

## Installation
#### Tracker installation
```bash
# 0. Let's say this repository is downloaded in ~/ATP

# 1. install anaconda3 for linux
# Download anaconda3 from: https://www.anaconda.com/
bash Anaconda3-<current-version>-Linux-x86_64.sh
# Let's say it is installed in ~/anaconda3

# 2. install our anaconda environment
# Download from: https://drive.google.com/open?id=11CuBpfbX7cJRryq-kJM3txILEG8DT9uT
unzip pytracking.zip
mv pytracking ~/anaconda3/envs

# 3. activate anaconda env
source activate pytracking

# 4. install python wrapper of Matlab
cd /PATH/TO/MATLAB_2018/extern/engines/python/
python setup.py build -b /tmp/matlab_engine_build install

# 5. compile operators
cd ~/ATP/ltr/external/PreciseRoIPooling/pytorch/prroi_pool
export PATH=/usr/local/cuda/bin/:$PATH
bash travis.sh

# 6. download pretrained models
# Download from: https://drive.google.com/open?id=1z9Tu0yo6YNSjtyLI9L0DvhayqoOcGyhC
mv networks ~/ATP/pytracking/networks
```

## Evaluation
```bash
# 0. follow VOT tutorials to setup the workspace
# refer to http://www.votchallenge.net/howto/workspace.html

# 1. build trax
cd /PATH/TO/vot-toolkit/native/trax && mkdir -p build && cd build
cmake .. && make

# 2. copy tracker_ATP.m into the VOT2019 workspace
cp ~/ATP/tracker_ATP.m /PATH/TO/VOT2019/WORKSPACE

# 3. set paths within tracker_ATP.m
# set </PATH/TO/ATP>
# set </PATH/TO/vot-toolkit/native/trax/build>

# 4. set tracker_load in run_experiments.m
# tracker = tracker_load('ATP');

# 5. activate pytracking environment
source activate pytracking

# 6. start evaluation, you should get EAO close to 0.393
matlab -r "run_experiments"
```

## Known issues
1. Warnings are safe to ignore.
2. Due to low running speed, you may frequently encounter the error of *Tracker execution interrupted: Did not receive response.* In this case, please restart the evaluation process.