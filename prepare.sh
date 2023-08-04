# to setup the environment for the experiments 
pip install virtualenv
/home/ubuntu/.local/bin/virtualenv -p python3.9.4 kd_env 
conda deactivate
source kd_env/bin/activate
which python
which pip

# install the requirements
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt


# download the dataset
mkdir ../imagenet_dataset
cd ../imagenet_dataset
cp ../vanillaKD/extract_ILSVRC.sh ./
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar; wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar; wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz; ./extract_ILSVRC.sh