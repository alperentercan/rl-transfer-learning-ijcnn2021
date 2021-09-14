import subprocess
import sys

iteration_dict={"CartPoleBulletPOScaled-v2":70003,"CartPoleBulletPO-v2":70003, "Acrobot-v1":125003, "AcrobotSparse-v1":125003}
def run_experiment(env, alg):
    outputs = []
    errs = []
    nIters = iteration_dict[env]
    print(nIters)
    for k in range(1,6):
        args=f"--experiment_repeat=1 --alg={alg} --env={env} --train_iter={nIters} --seed={k} --debug"            
        command = f"python ../src/main.py {args}"
        out = subprocess.run(command,shell=True,capture_output=True,text=True)
        outputs.append(out.stdout)
        print(outputs[-1])
        errs.append(out.stderr)
        
if __name__ == "__main__":
    env, alg = sys.argv[1], sys.argv[2]
    assert env in ["CartPoleBulletPOScaled-v2","CartPoleBulletPO-v2", "Acrobot-v1", "AcrobotSparse-v1"], "Invalid env, see source code"
    assert alg in ["pretraining", "dualtraining","vanilla"], "Invalid algorithm, see source code"
    run_experiment(env,alg)
