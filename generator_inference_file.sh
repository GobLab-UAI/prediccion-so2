#!/bin/bash
N_CLASSES="3"
BOOSTS="8"
ESTACIONES="k540"

for estacion in $ESTACIONES
do
    for n_class in $N_CLASSES
    do
        for boost in $BOOSTS
        do
            echo src/generator_inference_file.py $estacion $n_class $boost
            python src/generator_inference_file.py $estacion $n_class $boost;    
        done
    done
done

