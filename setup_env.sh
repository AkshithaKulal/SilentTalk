#!/bin/bash
set -e

git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2/huggingface_interface
source install.sh

mv IndicTransToolkit IndicTransToolkit_repo
pip install -e IndicTransToolkit_repo -q
pip install -U torchao -q
pip install sacrebleu -q
pip install transformers==4.53.2 -q

echo "Verifying import..."
python3 -c "from IndicTransToolkit import IndicProcessor, IndicDataCollator; print('Import successful')"
