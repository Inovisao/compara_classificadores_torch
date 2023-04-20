# Para instalar as dependências 
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

# É preciso instalar também um monte de pacotes para rodar o graficos.R
# Uma dica é tentar instalar o rstudio e abrir lá o graficos.R (ele já
# vai oferecer para instalar as dependências). Dando pau (o mais provável)
# terá que fazer umas instalações por favor também :-( 
# Na minha máquina (Ubuntu 22.04) precisei rodar estes comandos
# no terminal do Linux para conseguir tirar todos os erros que apareciam
# durante a instação dos pacotes no rstudio

sudo apt-get update
sudo apt-get install curl libxml2-dev libcurl4-openssl-dev cmake libfontconfig1-dev 

# O comando para instalar pacotes de dentro do rstudio é este aqui
install.packages("kableExtra")


