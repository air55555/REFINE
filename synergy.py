from utils import *

fl=[r"C:\Users\1\PycharmProjects\DPHSIRmy\out\250108_13-34-00\pred.hdr",
    r"C:\Users\1\PycharmProjects\DPHSIRmy\out\250108_13-35-27\pred.hdr"]
fl=[r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_max_56bands.hdr",
    r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_mean_56bands.hdr",
    r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_median_56bands.hdr",

    r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_std_56bands.hdr",
    r"C:\Users\1\PycharmProjects\hsi_cheese\reduced_hsi_min_56bands.hdr",
]
CMAP ='brg'
hsi_combined = synergize_hsi(read_hsi_files(fl),align=False)
save_hsi_as(hsi_combined,r".\synergy.hdr")
plt.imsave(r'.\synergy.png', generate_rgb(hsi_combined,1,5,20), cmap=CMAP)