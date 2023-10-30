
#!/bin/bash

aggregated_result_file="logs/adaicl_adaptive.txt"

model_name="gpt-neo"


for s in 0 
do
    printf "%6s\n" $e >> $aggregated_result_file
    for t in  "trec" "sst2"
    do

        for m in "ada_icl_default" 
        do
            printf "%10s\t" $m >> $aggregated_result_file
            python main_adaptive_phases.py  --phases 1  --few_shot 5 --task_name $t --selective_annotation_method $m --model_cache_dir "models" --data_cache_dir "datasets" --output_dir outputs_adaicl_thres --annotation_size 20 --model_name $model_name --seed $s --init "cluster"  --sample_k >> $aggregated_result_file
        done

        printf "\n" >> $aggregated_result_file    
    done
done
