#!/bin/sh

NUM_SEEDS=100
ALL_NUM_STATES=(10 25 50)
ALL_LEFT_PROBABILITY=(0.05 0.15 0.25 0.5)

ALPHA_VALUES_FILE=$(dirname $0)'/alpha_values.txt'
ETA_VALUES_FILE=$(dirname $0)'/eta_values.txt'
LAMBDA_VALUES_FILE=$(dirname $0)'/lambda_values.txt'

# assert command line arguments valid
if [ "$#" -gt "1" ]
    then
        echo 'usage: ./run.sh [RESULTS_DIR]'
        exit
    fi

# get folder name for results
if [ "$#" == "1" ]
    then
        RESULTS_DIR=$1
    else
        RESULTS_DIR=$(date +%Y-%m-%dT%H:%M:%S%z)
    fi

# build run tasks file
RUN_TASKS_FILE="$RESULTS_DIR"/'run_tasks.sh'
if [ ! -f $RUN_TASKS_FILE ]
    then
        for LEFT_PROBABILITY in "${ALL_LEFT_PROBABILITY[@]}"
            do
                for NUM_STATES in "${ALL_NUM_STATES[@]}"
                    do
                        NUM_STEPS=$((NUM_STATES * 100))
                        while read -r ALPHA
                            do
                                while read -r ETA
                                    do
                                        # add GTD tasks
                                        while read -r LAMBDA
                                            do
                                                DATA_DIR="$RESULTS_DIR"/"$LEFT_PROBABILITY"/"$NUM_STATES"/'GTD'/"$ALPHA"/"$ETA"/"$LAMBDA"
                                                mkdir -p "$DATA_DIR" 2>/dev/null
						echo "python $(dirname $0)/run_GTD.py $DATA_DIR $ALPHA $ETA $LAMBDA --leftprob $LEFT_PROBABILITY --numseeds $NUM_SEEDS --numstates $NUM_STATES --numsteps $NUM_STEPS" >> $RUN_TASKS_FILE
                                            done < $LAMBDA_VALUES_FILE

                                        # add LGGTD task
                                        DATA_DIR="$RESULTS_DIR"/"$LEFT_PROBABILITY"/"$NUM_STATES"/'LGGTD'/"$ALPHA"/"$ETA"
                                        mkdir -p "$DATA_DIR" 2>/dev/null
					echo "python $(dirname $0)/run_LGGTD.py $DATA_DIR $ALPHA $ETA --leftprob $LEFT_PROBABILITY --numseeds $NUM_SEEDS --numstates $NUM_STATES --numsteps $NUM_STEPS" >> $RUN_TASKS_FILE

                                        # add DLGGTD tasks
                                        DATA_DIR="$RESULTS_DIR"/"$LEFT_PROBABILITY"/"$NUM_STATES"/'DLGGTD'/"$ALPHA"/"$ETA"
                                        mkdir -p "$DATA_DIR" 2>/dev/null
					echo "python $(dirname $0)/run_DLGGTD.py $DATA_DIR $ALPHA $ETA $KAPPA --leftprob $LEFT_PROBABILITY --numseeds $NUM_SEEDS --numstates $NUM_STATES --numsteps $NUM_STEPS" >> $RUN_TASKS_FILE
                                    done < $ETA_VALUES_FILE
                            done < $ALPHA_VALUES_FILE
                    done
            done
    fi

# generate data
parallel :::: $RUN_TASKS_FILE

# plot results
for LEFT_PROBABILITY in "${ALL_LEFT_PROBABILITY[@]}"
    do
        for NUM_STATES in "${ALL_NUM_STATES[@]}"
            do
                PERFORMANCE_PLOT_FILE="$RESULTS_DIR"/'performance_'$LEFT_PROBABILITY'_'$NUM_STATES'.png'
                if [ ! -f $PERFORMANCE_PLOT_FILE ]
                    then
                        NUM_STEPS=$((NUM_STATES * 100))
			python $(dirname $0)'/plot_performance.py' $PERFORMANCE_PLOT_FILE $RESULTS_DIR $NUM_SEEDS $NUM_STATES $NUM_STEPS --leftprob $LEFT_PROBABILITY --alpha-values $ALPHA_VALUES_FILE --eta-values $ETA_VALUES_FILE --lambda-values $LAMBDA_VALUES_FILE
                    fi
            done
    done

# clean up
rm $RUN_TASKS_FILE

# zip results
zip -mqr $RESULTS_DIR.zip $RESULTS_DIR
