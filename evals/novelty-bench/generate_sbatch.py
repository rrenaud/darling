import argparse
import os
import textwrap

def main():
    parser = argparse.ArgumentParser(
        description="Generate an SBATCH script (stored in slurm_scripts/) with a given job name."
    )
    parser.add_argument(
        "job_name",
        help="Name of the job (used for #SBATCH --job-name, log filenames, and the output .batch filename)."
    )
    args = parser.parse_args()
    job_name = args.job_name

    # Fixed output directory and extension
    out_dir = "slurm_scripts"
    os.makedirs(out_dir, exist_ok=True)

    script_filename = os.path.join(out_dir, f"{job_name}.batch")

    template = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output=/home/tianjian/diverse_responses/logs/{job_name}.log
        #SBATCH --error=/home/tianjian/diverse_responses/logs/{job_name}.err
        #SBATCH --nodes=1
        #SBATCH --gpus=8
        #SBATCH --mem=64G
        #SBATCH --time=24:00:00
        #SBATCH --account=ram
        #SBATCH --qos=ram_high

        source /home/tianjian/.bashrc

        # Load the necessary modules
        conda activate vllm

        # write your command right here
        """)

    with open(script_filename, "w") as f:
        f.write(template + "\n")

    print(f"SBATCH script generated: {script_filename}")

if __name__ == "__main__":
    main()

