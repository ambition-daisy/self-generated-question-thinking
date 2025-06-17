cd /mnt/wx_feature/home/anglv/verl/
source ~/.bashrc
conda create -n selfq python=3.12
conda activate selfq
pip install -e .
conda install nvidia/label/cuda-12.4.0::cuda-nvcc
export CUDA_HOME=/miniconda3/envs/selfq
pip3 install flash-attn --no-build-isolation
pip install vllm