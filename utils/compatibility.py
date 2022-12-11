import glob
import os
import sys


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

def listdir(path):
    return list(listdir_nohidden(path))



if __name__ == '__main__':
    paths = sys.argv[1:]
    for path in paths:
        print(listdir(path))