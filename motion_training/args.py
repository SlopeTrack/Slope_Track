import argparse

def make_parser():
    parser = argparse.ArgumentParser("Motion")
    parser.add_argument("--model", default="lstm", type=str, help="mamba or lstm encoder")
    parser.add_argument("--option", default=5, type=int, help="option")
    parser.add_argument("--epochs", default=200, type=int, help="epochs")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--min-len", dest="min_len",  default=1, type=int, help="minimum length of sequence")
    parser.add_argument("--max-len", dest="max_len", default=30, type=int, help="maximum length of sequence")
    parser.add_argument("--batch-size", dest="batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--nll", dest="nll", default=False, action="store_true", help="Whether to use nll loss")
    parser.add_argument("--train", dest="train", default=False, action="store_true", help="train")
    parser.add_argument("--target-len", dest="target_len", default=1, type=int, help="length of trajectory prediction")
    return parser
