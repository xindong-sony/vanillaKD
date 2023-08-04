# to setup the environment for the experiments 
pip install virtualenv
virtualenv -p python3.9.4 kd_env 
source kd_env/bin/activate

# install the requirements
pip install -r requirements.txt


# download the dataset
mkdir ../imagenet_dataset
cd ../imagenet_dataset
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz

./extract_ILSVRC.sh