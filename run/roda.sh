# CLASSIFICADORES QUE VOCÊ QUER TESTAR
#arqs=(alexnet coat_tiny maxvit_rmlp_tiny_rw_256 vgg19 lambda_resnet26rpt_256 vit_relpos_base_patch32_plus_rpn_256 sebotnet33ts_256 lamhalobotnet50ts_256 swinv2_base_window16_256 convnext_base resnet18 ielt)
arqs=(resnet18 vit_relpos_base_patch32_plus_rpn_256)

# OPTIMIZADORES
#opt=(adam sgd adagrad lion sam)
opt=(sgd adagrad)

# LEARNING RATES
learning_rates=(0.001 0.0001)

cd ../src
for lr in "${learning_rates[@]}"
do
    for i in "${arqs[@]}"
    do
        for k in "${opt[@]}"
        do
            echo 'Running' ${lr} ' ' ${i} ' ' ${k} ' see results in folder ../results/'
            python3 main.py -a $i -o $k -r $1 -l $lr > >(tee -a ../results/${i}_${k}_${lr}.output) 2> >(tee ../results/error_log_${i}_${k}_${lr}.txt >&2)
            echo 'Finished running' ${lr} ' ' ${i} ' ' ${k}
        done
    done
done

cd ../run

