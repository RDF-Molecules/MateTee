#/bin/bash

echo "STARTING"

num_one=1
max_simultaneous=20

max=0
for (( i=0; i <= $max; ++i ))
do
    num_process_pre=$(ps -aux | grep TransE_complete_end2end.py | wc -l)
    num_current_processes=$((num_process_pre - num_one))
    
    if [ "$num_current_processes" -ge "$max_simultaneous" ] ; then
        echo "Unable to execute experiment $i. Waiting 1 min."
        sleep 60
        i=$((i - num_one))
    else
        now="$(date +'%m/%d/%Y %T')"
        echo "Executing experiment: $i , at $now ."
        nohup python -u TransE_complete_end2end.py $i >> transE_dump_$i.log &
        sleep 2
    fi    
done

echo "FINISHED"
