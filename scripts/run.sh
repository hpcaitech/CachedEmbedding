#!/bin/bash

# For Colossalai enabled recsys
bash scriptes/kaggle.sh

bash scriptes/avazu.sh

bash scriptes/terabyte.sh

# For TorchRec baseline
bash scriptes/torchrec_kaggle.sh

bash scriptes/torchrec_avazu.sh

bash scriptes/torchrec_terabyte.sh
