# Domestic cloud servers often have network issues with pip,
# so we need to pip install from tsinghua mirror
FROM hpcaitech/pytorch-cuda:1.12.0-11.3.0

RUN pip install --no-cache-dir petastorm[torch]==0.11.5

#install torchrec
RUN python3 -m pip install --no-cache-dir torchrec==0.2.0

#install torchrec
# RUN wget https://download.pytorch.org/whl/torchrec-0.1.1-py39-none-any.whl && \
#     python3 -m pip install --no-cache-dir torchrec-0.1.1-py39-none-any.whl && \
#     rm torchrec-0.1.1-py39-none-any.whl

# updated with hpcaitech version
RUN git clone https://github.com/hpcaitech/torchrec.git && cd torchrec && pip install .

# install colossalai
RUN git clone https://github.com/hpcaitech/ColossalAI.git && \
    cd ColossalAI/ && \
    python3 -m pip install -i --no-cache-dir -r requirements/requirements.txt && \
    python3 -m pip install -i --no-cache-dir . && \
    cd .. && \
    yes | rm -r ColossalAI/
