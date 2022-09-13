import argparse
from recbole.quick_start import run_kernel_calc_pytorch_logic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='OverRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_kernel_calc_pytorch_logic(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
