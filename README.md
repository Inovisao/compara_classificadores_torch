# compara_classificadores_torch

- Programa baseado no compara_classificadores_tf2, produzido pelo grupo Inovisão. O programa serve aos mesmos propósitos, tendo como diferencial o uso da biblioteca PyTorch, em vez do TensorFlow 2. 

## Rodando.

- Coloque as imagens separadas em uma pasta por classe dentro de ./data/all. Certifique-se de que os nomes dos arquivos não contêm espaços.
- Rode o script splitFolds.sh, passando o número de dobras como argumento -k (ex.: ./splitFolds.sh -k 10). Frequentemente se utilizam dez dobras.
- Selecione as redes (arquiteturas e otimizadores) a serem testadas em roda.sh.
- Altere os hiperparâmetros em hyperparameters.py.
- Se necessário, altere os hiperparâmetros dos otimizadores diretamente em optimizers.py.
- Rode o script rodaCruzada.sh, passando o número de dobras como argumento -k (ex.: ./rodaCruzada.sh -k 10).

## Instalação.
- Leia o arquivo install.txt

## Adicionando mais arquiteturas.

- Para adicionar uma nova arquitetura, defina uma função em architectures.py. Instancie a arquitetura e programe a alteração da primeira e da última camadas. Registre a nova arquitetura no dicionário que consta em arch_optim.py.
- Para adicionar um novo otimizador, defina uma função em optimizers.py. Os hiperparâmetros do otimizador devem estar declarados explicitamente, mesmo que o valor atribuído seja o valor padrão.
- Adicionei uma arquitetura, a IELT, que não é nem do torchvision nem do timm (embora usem o PyTorch na implementação). Acho que ela fornece um bom exemplo de como podemos aproveitar o código para testar arquiteturas diversas. Também adicionei dois otimizadores nas mesmas condições.

## Informações adicionais.

- Os resultados relativos à dobra em execução são colocados na pasta ./results. Os resultados por dobra são colocados na pasta ./resultsNfolds após a execução completa da dobra e os resultados finais (estatísticas, boxplots,etc) são colocados na pasta ./results_dl

## Troubleshooting.

- Certifique-se de que o ambiente correto está ativo.
- Algumas arquiteturas exigem configurações específicas, especialmente no que tange ao tamanho de imagem. Assim, por exemplo, a arquitetura coat_tiny exige imagens com dimensões exatamente iguais a (224, 224). Além disso, várias arquiteturas exigem que as imagens tenham um tamanho mínimo específico.
- Caso a memória da GPU seja insuficiente, diminua o tamanho do lote nos hiperparâmetros (batch size).
