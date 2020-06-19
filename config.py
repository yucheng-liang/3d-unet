import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # dataloader hyperparameters
    parser.add_argument("-r","--root_dir", type=str, default="./", help="root directory")
    parser.add_argument("-d","--data_dir", type=str, default="./training_data_0613/", help="path to mrsi and structural data")
    parser.add_argument("-s","--subjects", type=list, default=[126], help="list of subject numbers")
    parser.add_argument("-p","--patch_size", type=int, default=16, help="patch size in all 3 spatial dimensions")
    parser.add_argument("-m","--mode", type=str, default="test", help="either train or test")
    parser.add_argument("-n","--num_workers", type=int, default=0, help="number of workers for multiprocesses")
    parser.add_argument("--train_percentage", type=float, default=0.8, help="training data percentage, used only for 'train_test' mode")

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_iters", type=int, default=0, help="number of iterations")
    parser.add_argument("--resume_iters", type=int, default=30000, help="step the model resumes training from")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay coefficient for L2 regularization")
    parser.add_argument("--use_tensorboard", type=bool, default=False, help="Whether to activate tensorboard for visualizaion")
    parser.add_argument("--log_step", type=int, default=200, help="step to print out training information")
    parser.add_argument("--log_dir", type=str, default='./logger/', help="path to save logger")
    parser.add_argument("--model_save_step", type=int, default=1000, help="step to save model")
    parser.add_argument("--model_save_dir", type=str, default='./Checkpoint/', help="path to save model")
    parser.add_argument("--device_id", type=int, default=5, help="GPU device ID")

    # model hyperparameters
    parser.add_argument("--in_dim", type=int, default=143, help="input time dimension")
    parser.add_argument("--out_dim", type=int, default=1, help="output dimension")
    parser.add_argument("--num_filters", type=int, default=128, help="number filters")

    return parser.parse_args()
