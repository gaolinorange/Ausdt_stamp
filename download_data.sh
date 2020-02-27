
# download data
echo "dowload data and checkpoint"
mkdir data
curl -L -o elephhorse.mat https://www.dropbox.com/sh/xmgod9intwrjzrl/elephhorse.mat?dl=1
mv elephhorse.mat ./data/elephhorse.mat
curl -L -o checkpoint.zip https://www.dropbox.com/sh/6uk2mxfa9c9z5is/checkpoint.zip?dl=1
mv checkpoint.zip ./ckpt/checkpoint.zip
unzip ./ckpt/checkpoint.zip -d ./ckpt
