# Para instalar o compara_classificadores_torch. 
# Criar um ambiente conda.
conda create --name cla_torch

# Não testei com outras versões do pytorch.
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# O de sempre.
conda install -c conda-forge scikit-learn pandas matplotlib numpy seaborn opencv
pip install timm

pip install tifffile

pip install omnixai
