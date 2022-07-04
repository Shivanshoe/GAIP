import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--input_size', default=[3, 64, 64]) # change input size based on the preprocessed size of thumbnails
parser.add_argument('--beta1', default=0.5, help='Beta1 hyperparam for Adam optimizers')

parser.add_argument('--train_img_dir', type=str, default='../dataset/') # path to training imgs
parser.add_argument('--train_attr_path', type=str, default='../dataset/') # path to predictor and target variables pertaining to the training images (csv)
parser.add_argument('--test_img_dir', type=str, default='../dataset/') # path to testing imgs
parser.add_argument('--test_attr_path', type=str, default='../dataset/') # path to predictor and target variables pertaining to the testing images (csv)
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the dataset',
                    default=['Like_Count', 'Dislike_Count', 'Comment_Count', 'View_Count', 'Shares', 'Saves', 'Duration', 'Category_id', 'Title', 'Thumbnail', 'Description', 'Subscriber_Count', 'Channel_Age']) # To be modified based on the dataset
parser.add_argument('--datset_crop_size', type=int, default=178, help='crop size for the dataset') #hyperparamOptim
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')

parser.add_argument("--n_epochs", default=15, action="store", type=int, dest="n_epochs")
parser.add_argument("--z_size", default=128, action="store", type=int, dest="z_size")
parser.add_argument("--recon_level", default=3, action="store", type=int, dest="recon_level")
parser.add_argument("--lambda_mse", default=1e-6, action="store", type=float, dest="lambda_mse")
parser.add_argument("--lr", default=3e-4, action="store", type=float, dest="lr")
parser.add_argument("--decay_lr", default=0.75, action="store", type=float, dest="decay_lr")
parser.add_argument("--decay_mse", default=1, action="store", type=float, dest="decay_mse")
parser.add_argument("--decay_margin", default=1, action="store", type=float, dest="decay_margin")
parser.add_argument("--decay_equilibrium", default=1, action="store", type=float, dest="decay_equilibrium")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
