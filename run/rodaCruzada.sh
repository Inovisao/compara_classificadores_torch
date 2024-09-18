# RODA VALIDAÇÃO CRUZADA EM N DOBRAS

# É preciso rodar antes o script ./utils/splitFolds.sh

# Escolhe a GPU (0 ou 1 na Workstation do Inovisao)
export CUDA_VISIBLE_DEVICES=0

# IMPORTANTE: 3 dobras é muito pouco. Usei apenas para rodar mais apidamente um exemplo.
ndobras=5  
rodaPadrao=true
rodaSiamesa=true

# "treino" para apenas treinar, "teste" para apenas testar ou "completo" para executar ambos
procedimento="teste"

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

folds=()
for((i=1;i<=$ndobras;i+=1)); do folds+=("fold_${i}"); done

mkdir -p ../results_dl/           
rm -rf ../results_dl/*

if [ "$procedimento" != "teste" ]
then
   mkdir -p ../model_checkpoints/
   rm -rf ../model_checkpoints/*
fi

echo  'run,learning_rate,architecture,optimizer,precision,recall,fscore' > ../results_dl/results.csv

mkdir -p ${pastaResultados}
mkdir -p ${pastaTreino} 
mkdir -p ${pastaTeste}

rm -rf ${pastaDobrasResultados}/*
mkdir -p ${pastaDobrasResultados}

#Mudei este log de lugar, esta junto com os .output agora
#rm /tmp/deep_learning*log*

for Teste in "${folds[@]}"
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
    
   #break

done

if [ "$procedimento" != "treino" ]
then
   cd ../src
   Rscript ./graphics.R
   cd ../run
fi
 

