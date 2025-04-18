REM this could be run OUTSIDE the pycharm
print %PYTHONPATH%
SET PYTHONPATH=%PYTHONPATH%;D:\PycharmProjects\DPHSIRmy;D:\PycharmProjects\DPHSIRmy\etc
print %PYTHONPATH%
REM SET CUDA_VISIBLE_DEVICES = 2

S:\Users\LRS\miniconda3\envs\hirdiff\python.exe D:/PycharmProjects/DPHSIRmy/cli/main.py -t no_gt --device cuda -o timestamp -i d:\PycharmProjects\DPHSIRmy\input\synergy(256,256,88).mat -sf 2 -d grunet -s admm sisr -it 100
rem -o timestamp -i D:\PycharmProjects\DPHSIRmy\input\2024-09-13_08-04-13_white_circ_t_(256,256,50)_cropped.mat -sf 8   sisr -it 4