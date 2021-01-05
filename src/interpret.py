import argparse
import train
import data
import os


def main_cli():
    parser = argparse.ArgumentParser('CEDR model re-ranking')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--out_path', type=argparse.FileType('wt'))
    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--grad_layer_n', type=int, default=1)
    parser.add_argument('--masked_mode', type=float, default=None, help='masked mode')
    parser.add_argument('--with_proximity', type=int, default=0, help='with proximity')
    parser.add_argument('--gpunum', type=str, default="0", help='gpu number')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum

    model = train.MODEL_MAP[args.model]().cuda()
    dataset = data.read_datafiles(args.datafiles)
    run = data.read_run_dict(args.run)
    if args.model_weights is not None:
        model.load(args.model_weights.name)
        print(args.model_weights.name)
    #print("qweight=", model.qweight)
    #print(args.with_proximity)
    train.interpret_model(model, dataset, run, args.out_path.name, args.out_dir, args.grad_layer_n, desc='interpret', mode=args.masked_mode, with_proximity=args.with_proximity)


if __name__ == '__main__':
    main_cli()
