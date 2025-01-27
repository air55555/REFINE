print %PYTHONPATH%
SET PYTHONPATH=%PYTHONPATH%;D:\PycharmProjects\DPHSIRmy;D:\PycharmProjects\DPHSIRmy\etc
print %PYTHONPATH%
SET CUDA_VISIBLE_DEVICES = 0

S:\Users\LRS\miniconda3\envs\hirdiff\python.exe D:/PycharmProjects/DPHSIRmy/cli/main.py -o timestamp -i D:\PycharmProjects\DPHSIRmy\input\2024-09-13_08-04-13_white_circ_t_(256,256,50)_cropped.mat -sf 8   sisr -it 4