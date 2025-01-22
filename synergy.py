from utils import *

fl=[r"C:\Users\1\PycharmProjects\DPHSIRmy\out\250108_13-34-00\pred.hdr",
    r"C:\Users\1\PycharmProjects\DPHSIRmy\out\250108_13-35-27\pred.hdr"]
fl=[r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_max_88bands.hdr",
    r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_mean_88bands.hdr",
    r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_median_88bands.hdr",
    r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_std_88bands.hdr",
    r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_min_88bands.hdr",
]
250116_17-42-40
CMAP ='brg'
hsi_combined = synergize_hsi(read_hsi_files(fl),align=False)
#normalize
cube_min = hsi_combined.min()
cube_max = hsi_combined.max()
normalized_data = (hsi_combined - cube_min) / (cube_max - cube_min)
hsi_combined = normalized_data
save_hsi_as(hsi_combined, r".\synergy.hdr")
plt.imsave(r'.\synergy.png', generate_rgb(hsi_combined,1,5,20), cmap=CMAP)
envi_to_matlab("C:/Users/1\PycharmProjects\DPHSIRmy\synergy.hdr",r"C:\Users\1\PycharmProjects\DPHSIRmy\synergy"+str(hsi_combined.shape)+".mat")
#
