from utils import *
# Example usage
file_path = "log.txt"  # Replace with your actual log file path
log_content = read_log_file(file_path)

if log_content:
    print("Log file read successfully!")
    # You can now process the log content further
else:
    print("Failed to read the log file.")
log_entries = parse_log(log_content)

# Write the parsed entries to a CSV
write_to_csv(log_entries, 'log_results.csv')