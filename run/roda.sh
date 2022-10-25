# CLASSIFICADORES QUE VOCÊ QUER TESTAR
#arqs=(alexnet coat_tiny maxvit_rmlp_tiny_rw_256 vgg19)
arqs=(coat_tiny vgg19)

mkdir -p ../results
rm -rf ../results/*
mkdir -p ../results/history
mkdir -p ../results/matrix

# OPTIMIZADORES
#opt=(adam sgd)
opt=(adam sgd)

# LEARNING RATES
learning_rates=(0.01 0.001)

cd ../src
for lr in "${learning_rates[@]}"
do
    for i in "${arqs[@]}"
    do
        for k in "${opt[@]}"
        do
            echo 'Running' ${lr} ' ' ${i} ' ' ${k} ' see results in folder ../results/'
            python main.py -a $i -o $k -r $1 -l $lr > ../results/${i}_${k}_${lr}.output 2> ../results/error_log_${i}_${k}_${lr}.txt
        done
    done
done

cd ../run

