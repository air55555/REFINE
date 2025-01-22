import subprocess

from datetime import datetime

# Usage: main.py [OPTIONS] COMMAND [ARGS]...
#
# Options:
#   -i, --input_path TEXT      Path to input image/directory.  [required]
#   -o, --output_path TEXT     Path to output image/directory.  [default: tmp]
#   -d, --denoiser TEXT        Denoiser type.  [default: grunet]
#   denoiser, choices             [qrnn3d, qrnn3d_map, grunettv, grunet, drunet, ircnn, ffdnet, ffdnet3d, tv]
#   -dp, --denoiser_path TEXT  Path to denoiser model.  [default:
#                              ..\models\grunet.pth]
#   -s, --solver [admm|hqs]    Solver type.  [default: admm]
#   -sf INTEGER                scaling factor (nor requared)   [default: 2]
#   -t TEXT                    Type of infer - with gt or no_gt - real superres
#                              task   [default: with_gt]
#   --device TEXT              Device to use.  [default: cuda]
#   --help                     Show this message and exit.
#
# Commands:
#   cs
#   deblur
#   inpaint
#   misr
#   sisr
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
        r"D:\PycharmProjects\DPHSIRmy\cli\main.py"
        #"C:/Users/1/PycharmProjects/DPHSIRmy/cli/main.py",  # Replace with your script filenames
        #"C:/Users/1/PycharmProjects/DPHSIRmy/cli/main.py",
    ]
    # Common parameters for all scripts
    base_params = r"-o timestamp -i C:\Users\1\PycharmProjects\DPHSIRmy\input" \
                  r"\white_circ64,64,10)_cropped.mat -t no_gt --device cpu "
    base_params = r"-o timestamp -i C:\Users\1\PycharmProjects\DPHSIRmy\input" \
                  r"\31bands_(512,512,31)_downsampled_sf8.mat" \
                  r" -t no_gt --device cpu "
    base_params = r"-i D:\PycharmProjects\DPHSIRmy\input\synergy(256,256,88).mat " \
                  r"-sf 4 -d tv" \
                  r"    -o timestamp sisr "
                  #sisr


    # Parameter variants
    param_variants = [
        { "it": 1 },
        { "it": 2 },
    ]
    param_variants = [{"it": i} for i in range(1, 50)]

    with open(r'..\log.txt', "a") as file:
        batch = datetime.now().strftime("%y%m%d_%H-%M-%S")
        file.write(f"---batch{batch}\n")
    # Run the scripts with the specified parameters
    run_scripts(script_paths, base_params, param_variants)
