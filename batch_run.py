import subprocess

from datetime import datetime


#value = "-o timestamp -i C:\Users\1\PycharmProjects\DPHSIRmy\input\white_circ64,64,10)_cropped.mat  -sf 2  -t no_gt --device cpu   sisr -it 1" / >
no_prefix_keys = ["sisr", "misr", "another_key"]

def run_scripts(script_paths, base_params, param_variants):
    """
    Run multiple Python scripts with similar parameters.

    Parameters:
        script_paths (list): List of Python script paths to execute.
        base_params (str): Base parameters (common to all scripts).
        param_variants (list): List of parameter dictionaries to customize each run.
    """
    for script_path in script_paths:
        for variant in param_variants:
            params=[]
            for key, value in variant.items():
                if key in no_prefix_keys:  # Special case for commands
                    params.append(f"{value}")
                else:
                    params.append(f"-{key} {value}")
            params = ' '.join(params)
            full_command = f"python {script_path} {base_params} {params}"
            print(f"Executing: {full_command}")
            try:
                subprocess.run(full_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script {script_path} with parameters {params}: {e}")


# Example usage
if __name__ == "__main__":
    # Paths to Python scripts
    script_paths = [
        "C:/Users/1/PycharmProjects/DPHSIRmy/cli/main.py",  # Replace with your script filenames
        #"C:/Users/1/PycharmProjects/DPHSIRmy/cli/main.py",
    ]
    # Common parameters for all scripts
    base_params = r"-o timestamp -i C:\Users\1\PycharmProjects\DPHSIRmy\input" \
                  r"\white_circ64,64,10)_cropped.mat -t no_gt --device cpu "
    base_params = r"-o timestamp -i C:\Users\1\PycharmProjects\DPHSIRmy\input" \
                  r"\31bands_(512,512,31)_downsampled_sf8.mat" \
                  r" -t no_gt --device cpu "

    # Parameter variants
    param_variants = [
        {"sf": 2, "sisr": "sisr" , "it": 1 },
        {"sf": 2, "misr": "misr" , "it": 1 },
    ]
    with open(r'..\log.txt', "a") as file:
        batch = datetime.now().strftime("%y%m%d_%H-%M-%S")
        file.write(f"---batch{batch}\n")
    # Run the scripts with the specified parameters
    run_scripts(script_paths, base_params, param_variants)
