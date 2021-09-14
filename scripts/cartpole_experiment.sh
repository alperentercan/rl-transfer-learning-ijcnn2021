N_SEEDS=5
k=1
while [ $k -le  $N_SEEDS ]
do
    args="--experiment_repeat=1 --alg=vanilla --env=CartPoleBulletPOScaled-v2 --train_iter=70003 --seed=${k} --debug --output=../script-output-"
    python ../src/main.py ${args}
    echo "Vanilla experiment is run with seed ${k}"
    k=$(( $k + 1 ))    
done

k=1
while [ $k -le  $N_SEEDS ]
do
    args="--experiment_repeat=1 --alg=pretraining --env=CartPoleBulletPOScaled-v2 --train_iter=70003 --seed=${k} --debug --output=../script-output-"
    python ../src/main.py ${args}
    echo "Pretraining experiment is run with seed ${k}"
    k=$(( $k + 1 ))    
done

k=1
while [ $k -le  $N_SEEDS ]
do
    args="--experiment_repeat=1 --alg=dualtraining --env=CartPoleBulletPOScaled-v2 --train_iter=70003 --seed=${k} --debug --output=../script-output-"
    python ../src/main.py ${args}
    echo "Dualtraining experiment is run with seed ${k}"
    k=$(( $k + 1 ))    
done

python cartpole_plot.py
