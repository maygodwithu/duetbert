import argparse
import train
import data
import os
import torch


def main_cli():
    MODEL_MAP = train.modeling.MODEL_MAP
    parser = argparse.ArgumentParser('CEDR model re-ranking')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--submodel1', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel2', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel3', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel4', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel5', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel6', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel7', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel8', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--run', type=argparse.FileType('rt'))
    #parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--out_path', type=argparse.FileType('wt'))
    parser.add_argument('--gpunum', type=str, default="0", help='gup number')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    #setRandomSeed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum

    print("GPU count=", torch.cuda.device_count())

    #model = train.MODEL_MAP[args.model]().cuda()
    if(args.model.startswith('duet')):
        model = MODEL_MAP[args.model]( 
                    MODEL_MAP[args.submodel1](), 
                    MODEL_MAP[args.submodel2]()
                )
    elif(args.model.startswith('trio')):
        model = MODEL_MAP[args.model]( 
                    MODEL_MAP[args.submodel1](), 
                    MODEL_MAP[args.submodel2](),
                    MODEL_MAP[args.submodel3]()
                )
    elif(args.model.startswith('quad')):
        model = MODEL_MAP[args.model]( 
                    MODEL_MAP[args.submodel1](), 
                    MODEL_MAP[args.submodel2](),
                    MODEL_MAP[args.submodel3](),
                    MODEL_MAP[args.submodel4]()
                )
    elif(args.model.startswith('octo')):
        model = MODEL_MAP[args.model]( 
                    MODEL_MAP[args.submodel1](), 
                    MODEL_MAP[args.submodel2](),
                    MODEL_MAP[args.submodel3](),
                    MODEL_MAP[args.submodel4](),
                    MODEL_MAP[args.submodel5](), 
                    MODEL_MAP[args.submodel6](),
                    MODEL_MAP[args.submodel7](),
                    MODEL_MAP[args.submodel8]()
                )
    else:
        model = MODEL_MAP[args.model]().cuda()

    dataset = data.read_datafiles(args.datafiles)
    run = data.read_run_dict(args.run)

    if(args.model_weights is not None):
        wts = args.model_weights.split(',')
        if(len(wts) == 1):
            model.load(wts[0])
        elif(len(wts) == 2):
            model.load_duet(wts[0], wts[1])
        elif(len(wts) == 4):
            model.load_quad(wts[0], wts[1], wts[2], wts[3])
        elif(len(wts) == 6):
            model.load_hexa(wts[0], wts[1], wts[2], wts[3], wts[4], wts[5])
        elif(len(wts) == 8):
            model.load_octo(wts[0], wts[1], wts[2], wts[3], wts[4], wts[5], wts[6], wts[7])

    train.run_model(model, dataset, run, args.out_path.name, desc='rerank')


if __name__ == '__main__':
    main_cli()
