
# Update pip
pip install -U pip

# Create a virtual environment 
virtualenv --no-site-packages -p python3 ~/venv

# Activate the virtual environment
source ~/venv/bin/activate

# download pymesh
mkdir package
wget https://github.com/PyMesh/PyMesh/releases/download/v0.2.1/pymesh2-0.2.1-cp36-cp36m-linux_x86_64.whl
mv pymesh2-0.2.1-cp36-cp36m-linux_x86_64.whl ./package/pymesh2-0.2.1-cp36-cp36m-linux_x86_64.whl

# install tensorflow
pip install tensorflow-gpu==1.12.0

# install necessary library
pip install h5py==2.8.0
pip install pickleshare==0.7.5
pip install scipy==1.3.0
pip install scikit-learn==0.21.2
pip install ./package/pymesh2-0.2.1-cp36-cp36m-linux_x86_64.whl
pip install numpy==1.16.0

echo "installing sucessfully"

# download data
echo "dowload data and checkpoint"
curl -L -o elephhorse.mat https://www.dropbox.com/sh/xmgod9intwrjzrl/elephhorse.mat?dl=1
mv elephhorse.mat ./data/elephhorse.mat
curl -L -o checkpoint.zip https://www.dropbox.com/sh/6uk2mxfa9c9z5is/checkpoint.zip?dl=1
mv checkpoint.zip ./ckpt/checkpoint.zip
unzip ./ckpt/checkpoint.zip -d ./ckpt
