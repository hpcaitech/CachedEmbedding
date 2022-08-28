#!/bin/bash

# For Colossalai enabled recsys
bash kaggle.sh

bash avazu.sh

bash terabyte.sh

# For TorchRec baseline
bash torchrec_kaggle.sh

bash torchrec_avazu.sh

bash torchrec_terabyte.sh
