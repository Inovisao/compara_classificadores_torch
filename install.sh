# Para instalar as dependências do compara_regresores_torch
# COPIE E COLE OS COMANDOS ABAIXO NO TERMINAL
# NÃO FUNCIONA RODANDO O SCRIPT (AINDA).
# Criar um ambiente conda.
conda create --name cla_torch
conda activate cla_torch

# Não testei com outras versões do pytorch.
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# O de sempre.
conda install -c conda-forge scikit-learn pandas matplotlib numpy seaborn opencv
pip install timm tifffile omnixai

