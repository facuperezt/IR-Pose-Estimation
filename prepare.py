import os

if __name__ == '__main__':
    """
    Simply runs all python scripts in one line (after obj classification).
    Has to be called from the py37 env, and it opens the py27 env in the middle.
    Yes, this is most definitely bad practice, and an absolute safety hazard.

    BEFORE RUNNING THE SCRIPT:
        2. run 'python single_spot_table/obj_geo_based_classification.py' and follow the instructions

    """
    # Fetch Dataset
    os.system('python model_splitter.py 2 -t 1 10101')
    # Label Dataset
    os.system('python single_spot_table/obj_geo_based_classification.py')
    # Make Lookup Table
    os.system('python lut.py --label=HFD --fast_sampling')
    # Preprocess Training Data
    os.system('python train.py -p')
    # Preprocess Testing Data
    os.system('python test.py -p')