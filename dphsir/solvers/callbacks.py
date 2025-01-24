from tqdm import tqdm
class ProgressBar:
    def __init__(self, total):

        self.pbar = tqdm(total=total, dynamic_ncols=True)
        self.log_file = r'..\log.txt'
        self.iter_times = []

    def __call__(self, **context):
        self.pbar.update()
        iter_time = 1/(self.pbar.format_dict['rate']) # calculate it/s as 1/rate
        self.iter_times.append(iter_time)

    def all_close(self):
        """
        Closes the progress bar and logs the final progress state to a file.
        """

        self._log_to_file()

    def _log_to_file(self):
        """
        Logs the progress details to the specified file.
        """
        #str =tqdm(total=total)
        with open(self.log_file, "a") as f:
            f.write("Iteration times (seconds):\n")
            f.write(", ".join(f"{t:.4f}" for t in self.iter_times) + "\n")

class GatherIntermediates:
    def __init__(self, filter):
        self.intermediates = []
        self.filter = filter
    
    def __call__(self, **context):
        saved = self.filter(context)
        self.intermediates.append(saved)