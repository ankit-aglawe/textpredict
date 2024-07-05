import os


def crawl_directory_and_write_files(base_dir, output_file):
    """
    Crawl through the base directory, find all Python files, and write their paths and contents to the output file.

    Args:
        base_dir (str): The base directory to crawl through.
        output_file (str): The output text file to write the paths and contents of the Python files.
    """
    with open(output_file, "w") as out_file:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    out_file.write(f"Path: {file_path}\n")
                    out_file.write("Code:\n")
                    with open(file_path, "r") as py_file:
                        out_file.write(py_file.read())
                    out_file.write("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    # Set the base directory you want to crawl through and the output file name
    base_directory = "/home/machinelearning/Desktop/WORK/PY/textpredict/textpredict"  # Replace with your base directory
    output_filename = "all_python_files.txt"

    crawl_directory_and_write_files(base_directory, output_filename)
    print(f"All Python files and their contents have been written to {output_filename}")
