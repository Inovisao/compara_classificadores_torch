# RODA VALIDAÇÃO CRUZADA EM N DOBRAS

# É preciso rodar antes o script ./utils/splitFolds.sh

# Escolhe a GPU (0 ou 1 na Workstation do Inovisao)
export CUDA_VISIBLE_DEVICES=0

# IMPORTANTE: 3 dobras é muito pouco. Usei apenas para rodar mais apidamente um exemplo.
ndobras=5
rodaPadrao=true
rodaSiamesa=true

# "treino" para apenas treinar, "teste" para apenas testar ou "completo" para executar ambos
procedimento="completo"

# Verifica se o usuário passou como parâmetro
# o número de dobras (E.g.: ./rodaCruzada.sh -k 5)
while getopts "k:" flag; 
do
   case "${flag}" in
      k) ndobras=${OPTARG}
         ;;
   esac
done

# Nomes das pastas onde ficarão os resultados para cada dobra
pastaDobrasImagens="../data/dobras"
pastaTreino="../data/train"
pastaTeste="../data/test"
pastaResultados="../results"
pastaDobrasResultados="../resultsNfolds"
# Local do arquivo de dobras completas
dobrasCompletasArq="./dobrasCompletasArq.txt"

folds=()
for((i=1;i<=$ndobras;i+=1)); do folds+=("fold_${i}"); done

# Cria pastas de resultados
mkdir -p ../results_dl/
mkdir -p ${pastaResultados}
mkdir -p ${pastaTreino} 
mkdir -p ${pastaTeste}
mkdir -p ${pastaDobrasResultados}

# Cria dobrasCompletasArq caso não exista
if [ ! -f $dobrasCompletasArq ]; then
   touch $dobrasCompletasArq
fi

# Coloca as dobras existentes em um array
dobrasCompletas=($(cat $dobrasCompletasArq))

# Registra dobras restantes
dobrasRestantes=()
for fold in "${folds[@]}"; do
   if [[ ! " ${dobrasCompletas[@]} " =~ " ${fold} " ]]; then
      dobrasRestantes+=("${fold}")
   fi
done

# Confere se o usuário quer continuar os testes
if [ ${#dobrasCompletas[@]} -gt 0 ]; then
   echo "${#dobrasCompletas[@]} testes de ${ndobras} executados."
   read -p "Deseja continuar? (s/n):" choice
   if [[ "$choice" == "n" || "$choice" == "N" ]]; then
      # Apagar dobras completadas e reiniciar
      echo "Reiniciando testes..."
      rm -rf ${pastaTreino}/* ${pastaTeste}/* ${pastaResultados}/* ${pastaDobrasResultados}/*
      rm -rf ../results_dl/*
      echo  'run,learning_rate,architecture,optimizer,precision,recall,fscore' > ../results_dl/results.csv
      rm $dobrasCompletasArq
      touch $dobrasCompletasArq
      # Reinicia dobrasRestantes e Completas
      dobrasRestantes=("${folds[@]}")
      dobrasCompletas=()
   else
      echo "Continuando testes nas dobras restantes..."
      # Atualiza dobrasCompletas e reinicia
      dobrasCompletas=($(cat "$dobrasCompletasArq"))
      dobrasRestantes=()
      for fold in "${folds[@]}"; do
         if [[ ! " ${dobrasCompletas[@]} " =~ " ${fold}" ]]; then
            dobrasRestantes+=("${fold}")
         fi
      done
   fi
else
   echo "Iniciando novo teste..."
fi

if [ "$procedimento" != "teste" ]
then
   mkdir -p ../model_checkpoints/
   rm -rf ../model_checkpoints/*
fi

#Mudei este log de lugar, esta junto com os .output agora
#rm /tmp/deep_learning*log*

for Teste in "${dobrasRestantes[@]}"
do
  
   echo 'Preparing test on' ${Teste} '...'
   rm -rf ${pastaTreino}/*
   rm -rf ${pastaTeste}/*
   rm -rf ${pastaResultados}/*
   
   cp -R ${pastaDobrasImagens}/${Teste}/* ${pastaTeste} 

   for outro in "${folds[@]}" 
   do 
      if [ ${outro} != ${Teste} ] 
      then
         echo 'Adding to train' ${outro} 
         cp -R ${pastaDobrasImagens}/${outro}/* ${pastaTreino} 
      fi   
   done
   
   run=${Teste#*_}

   mkdir -p ../results
   rm -rf ../results/*
   mkdir -p ../results/history
   mkdir -p ../results/matrix

   bash ./roda.sh $run $rodaPadrao $rodaSiamesa $procedimento
  
   mkdir -p ${pastaDobrasResultados}/${Teste}
   mv ${pastaResultados}/* ${pastaDobrasResultados}/${Teste}   
   
   echo $Teste >> $dobrasCompletasArq
   #break
done

if [ "$procedimento" != "treino" ]
then
   cd ../src
   Rscript ./graphics.R
   cd ../run
fi
 

