import os

if __name__ == '__main__':
    """
    Simply runs all python scripts in one line (after obj classification).
    Has to be called from the py37 env, and it opens the py27 env in the middle.
    Yes, this is most definitely bad practice, and an absolute safety hazard.

    BEFORE RUNNING THE SCRIPT:
        1. run 'python model_splitter.py 2 -t 1 10101' to load a small dataset
        2. run 'python single_spot_table/obj_geo_based_classification.py' and follow the instructions

    """
    # Make Lookup Table
    os.system('python lut.py --label=HFD --fast_sampling')
    # Preprocess Training Data
    os.system('python train.py -p')
    # Preprocess Testing Data
    os.system('python test.py -p')
    # Activate Python2 environment
    os.system('conda activate py27')
    # Train the model
    os.system('python train.py -t --max_epoch 5')
    # Run inference
    os.system('python test.py -i')