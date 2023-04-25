<<<<<<< HEAD:install.txt
O próximo que for instalar, antes de rodar o que tem aqui, tente usar o seguinte comando:

conda env create -f environment.yml

O objetivo é que ele instale o conteúdo de environment.yml automaticamente e reduza o problema com versões.
Se funcionar, avise neste arquivo.
Se não funcionar, tente encontrar o que deu errado.
Se não funcionar, uma possibilidade é que seja um problema com o driver da GPU. Estou usando a versão 510.108.03, com CUDA 11.6.

# Para instalar as dependências do compara_regresores_torch
=======
# Para instalar as dependências 
>>>>>>> 0a96e5a1a30feb2b3476c8e6d255a34d47f93214:install.sh
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


