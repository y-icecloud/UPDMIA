import argparse
from run_detect_feature import run_detect_feature
from run_test_classifier import run_test_classifier
from run_train_feature import run_train_feature
from run_train_classifier import run_train_classifier
from run_test_feature import  run_test_feature



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fea_test_dir', default='./datasets/feature/test', required=False)
    parser.add_argument('--fea_train_dir', default='./datasets/feature/train', required=False)
    parser.add_argument('--fea_model_path', default='./best_model.pth', required=False)

    parser.add_argument('--cla_test_dir', default='./datasets/classification/test', required=False)
    parser.add_argument('--cla_train_dir',default='./datasets/classification/train', required=False)
    parser.add_argument('--cla_model_path', required=False)

    parser.add_argument('--f', default=True, required=False)
    parser.add_argument('--detect', default=False,required=False)
    parser.add_argument('--save_path', default='./',required=False)

    args = parser.parse_args()
    best_loss = float("inf")
    epochs = 100
    for i in range(epochs):
        print(f'epoch[{i+1}/{epochs}]')
        if args.f:
            if not args.detect:
                run_train_feature(args)
                best_loss = run_test_feature(args,best_loss)
            else:
                run_detect_feature(args)
        else:
            if not args.detect:
                run_train_classifier(args)
                run_test_classifier(args)

