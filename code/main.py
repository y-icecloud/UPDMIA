import argparse
import os
from TrainTestFunction import run_test_classification, run_train_classification
from DetectFunction import run_detect
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cla_test_dir', default='./datasets/LungCancerSlideDataset/test', required=False)
    parser.add_argument('--cla_train_dir',default='./datasets/LungCancerSlideDataset/train', required=False)
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--save_interval', default =5)
    parser.add_argument('--save_dir', default = './working/')
    parser.add_argument('--category', default='Slide')
    parser.add_argument('--detect', default=True,required=False)
    
    
    args = parser.parse_args()
    if args.detect:
        run_detect(args)
    else:
        run_train_classification(args)
        run_test_classification(args)