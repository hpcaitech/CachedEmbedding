FROM nvcr.io/nvidia/merlin/merlin-pytorch-training:latest

#install torch
RUN conda update -n base conda && \
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch  

#install fbgemm_gpu
RUN python3 -m pip install --no-cache-dir fbgemm_gpu==0.1.1

#install torchrec
RUN wget https://download.pytorch.org/whl/torchrec-0.1.1-py38-none-any.whl && \
    python3 -m pip install --no-cache-dir torchrec-0.1.1-py38-none-any.whl && \
    rm torchrec-0.1.1-py38-none-any.whl

# install colossalai
RUN git clone https://github.com/hpcaitech/ColossalAI.git && \
    cd ColossalAI/ && \
    python3 -m pip install --no-cache-dir -r requirements/requirements.txt && \
    python3 -m pip install --no-cache-dir . && \
    cd .. && \
    yes | rm -r ColossalAI/

