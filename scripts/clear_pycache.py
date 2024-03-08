from glob import glob
from shutil import rmtree

if __name__ == "__main__":
    for pycache in glob("**/__pycache__", recursive=True):
        print(f"Removing {pycache}")
        rmtree(pycache)
    print("Done!")
