FROM hpcaitech/pytorch-cuda:1.11.0-11.3.0

#install fbgemm_gpu
RUN python3 -m pip install --no-cache-dir fbgemm_gpu==0.1.1

#install torchrec
RUN wget https://download.pytorch.org/whl/torchrec-0.1.1-py39-none-any.whl && \
    python3 -m pip install --no-cache-dir torchrec-0.1.1-py39-none-any.whl && \
    rm torchrec-0.1.1-py39-none-any.whl

# install colossalai
RUN git clone https://github.com/hpcaitech/ColossalAI.git && \
    cd ColossalAI/ && \
    python3 -m pip install --no-cache-dir -r requirements/requirements.txt && \
    python3 -m pip install  --no-cache-dir . && \
    cd .. && \
    yes | rm -r ColossalAI/

RUN pip install --no-cache-dir petastorm[torch]
