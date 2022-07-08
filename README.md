# tensorflow-config-linux
Tutorial on how to configure tensorflow with GPU on linux mint
## Install environment
### Install anaconda
Download .sh file [here](https://www.anaconda.com/products/distribution)  
install with `bash name.sh`  

### Install spyder
`conda install spyder`

### Modifications in .bashrc
export PATH="/home/{user}/anaconda3/bin:$PATH" (replace user with your username)  

`python --version` command in terminal should return the last python version  

Test your installation by typing `spyder` in the terminal (should open spyder IDE)  

### Install OpenCV
`pip install opencv-python`

## GPU Configuration
### Install Cuda
Linux Mint 20.3 is based on Ubuntu 20.04, usefull to know which version take  
Follow steps [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

### Install cuDNN
Download the deb package [here](https://developer.nvidia.com/rdp/cudnn-download)  

May need to add .deb at the end of the file in order to do the next steps.  

Follow the steps [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb)  

Verify the installation by following the steps [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify)  
If the test fails to compile, try to install the required libs  
I had to do `sudo apt install libfreeimage-dev`

## Tensorflow Configuration
### Install tensorflow
Use the right url depending on your python version (`python --version`)  
More info [here](https://www.tensorflow.org/install/pip#package-location)  

`pip install https://storage.googleapis.com/tensorflow/linux/gpu/[name].whl`

### Test with a basic denoising AI
Load the mnist_denoising_autoencoder.py file in spyder and run the script
