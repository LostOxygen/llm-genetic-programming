"""main hook for the project"""
import os
import datetime
import getpass
import torch
import argparse

from utils.colors import TColors

def main(device: str) -> None:
    """
    Main function to run the project.

    Parameters:
        device (str): device to run the computations on (cpu, cuda, mps).
    """

    # set the devices correctly
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device(device)
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device(device)
    else:
        print(f"{TColors.WARNING}Warning{TColors.ENDC}: Device {device} is not available. " \
               "Setting device to CPU instead.")
        device = torch.device("cpu")

    print("\n"+"#"*os.get_terminal_size().columns)
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: " + \
          str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: " \
          f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and " \
          f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if torch.cuda.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: " \
              f"{torch.cuda.mem_get_info()[1] // 1024**2} MB")
    print("\n"+"#"*os.get_terminal_size().columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-genetic-programming")
    parser.add_argument("--device", "-d", type=str, default="cuda",
                        help="specifies the device to run the computations on (cpu, cuda, mps)")
    args = parser.parse_args()
    main(**vars(args))
