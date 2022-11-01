# Determined Image
FROM determinedai/environments:cuda-11.3-pytorch-1.10-lightning-1.5-tf-2.8-gpu-83dbcaa

# Custom Configuration
RUN eval "$(conda shell.bash hook)" && \
    conda activate base && \
    conda install -y -c conda-forge -c pytorch -c fvcore -c pytorch3d \
        python=3.8 \
        pytorch \
        torchvision \
        cudatoolkit \
        opencv=4.4.0 \
        tqdm=4.50.2 \
        scipy=1.5.2 \
        fvcore=0.1.2 \
        pytorch3d && \
    pip install \
        pytools==2020.4 \
        kornia==0.3.2 \
        path==15.0.0 \
        protobuf==3.13.0 \
        tensorboardx==2.1 