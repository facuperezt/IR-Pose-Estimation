import os

if __name__ == '__main__':
    # Train the model
    os.system('python train.py -t --max_epoch 5')
    # Run inference
    os.system('python test.py -i')