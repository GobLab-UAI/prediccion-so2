#!/bin/bash
N_CLASSES="2 3 4"
BOOSTS="8 16 24"
ESTACIONES="k540"
FLAG_FILE_TEST=true

for estacion in $ESTACIONES
do
    for n_class in $N_CLASSES
    do
        for boost in $BOOSTS
        do
            echo src/experiment_train.py $estacion $n_class $boost
            python src/experiment_train.py $estacion $n_class $boost > ${estacion}_${n_class}_${boost}_train.txt
            
            echo src/experiment_test.py $estacion $n_class $boost
            python src/experiment_test.py $estacion $n_class $boost > ${estacion}_${n_class}_${boost}_test.txt
            
            if $FLAG;
            then
                echo src/generator_inference_filetest.py $estacion $n_class $boost
                python src/generator_inference_filetest.py $estacion $n_class $boost;
            fi    
        done
    done
    FLAG_FILE_TEST=false
done

