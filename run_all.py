import subprocess
import wandb
import sys

MAX_JOBS = 4

layer_sizes = ["", "512", "512,512", "1024,1024"]
model_names = ["avsolatorio/GIST-small-Embedding-v0", "Alibaba-NLP/gte-base-en-v1.5", "Alibaba-NLP/gte-large-en-v1.5", "google/mobilebert-uncased"]

n_jobs_started = 0
# for use_narrations in [False, True]:
for use_narrations in [True]:
    for layer_size in layer_sizes:
        for model_name in model_names:
            print(f"Config: layer_size={layer_size}, model_name={model_name}, use_narrations={use_narrations}")
            if 'gte-large' in model_name and use_narrations:
                print("Skipping large model with narrations (too expensive)")
                continue

            sbatch_options = ["-c 14", "--time=24:00:00"]
            if 'gte-base' in model_name and use_narrations:
                sbatch_options += ["--partition=g48", "--gres=gpu:1"]
            elif 'gte-base' in model_name:
                sbatch_options += ["--partition=g24", "--gres=gpu:1"]
            elif 'gte-large' in model_name:
                sbatch_options += ["--partition=g48", "--gres=gpu:1"]
            elif 'mobilebert' in model_name:
                sbatch_options += ["--partition=g24", "--gres=gpu:1"]
            
            batch_size_arg = ""
            if 'gte-large' in model_name:
                batch_size_arg = "--batch_size=32"
            if use_narrations:
                batch_size_arg = "--batch_size=32"
            if use_narrations and 'gte' in model_name:
                batch_size_arg = "--batch_size=16"

            # Check if run exists already
            run_exists = False
            for run in wandb.Api().runs("rug-minds/parse-ego4d"):
                layer_cond = ",".join(map(str, run.config.get("layer_sizes"))) == layer_size
                name_cond = run.config.get("model_name") == model_name
                state_cond = run.state in ["finished", "running"]
                narr_cond = run.config.get("use_narrations") == use_narrations
                run_exists = all([layer_cond, name_cond, state_cond, narr_cond])
                if run_exists:
                    break

            if run_exists:
                continue
            elif n_jobs_started >= MAX_JOBS:
                print("Max number of jobs started, quitting...\n")
                sys.exit(0)
            else:
                print(f"Launching run ({batch_size_arg} {sbatch_options})")
                subprocess.run([
                    "sbatch", *sbatch_options, 
                    "run_experiment.sh", 
                    f"--layer_sizes={layer_size}", 
                    f"--model_name={model_name}",
                    "--use_narrations" if use_narrations else "",
                    batch_size_arg,
                ])
                n_jobs_started += 1
                print()
                if n_jobs_started >= MAX_JOBS:
                    print("Max number of jobs started, quitting...\n")
                    sys.exit(0)

# for run in wandb.Api().runs("rug-minds/parse-ego4d"):
#     print(run.config.get("layer_sizes"), run.config.get("model_name"), run.state)