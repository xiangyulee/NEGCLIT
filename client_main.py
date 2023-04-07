import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from experiment.client import *
import torch
from util.name_match import dataset_class_num,online_dataset_name,model_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument('--seed', dest='seed', default=1, type=int,
                        help='Random seed')

    parser.add_argument("--ip", type=str,default='127.0.0.1')
    parser.add_argument("--port", type=int,default=3001)
    parser.add_argument("--world_size", type=int,default=2)
    parser.add_argument("--rank", type=int,default=1)
    parser.add_argument("--ethernet", type=str, default=None)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_class", type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', default='resnet', type=str, metavar='MODEL',
                        help='whole model:NE+NG(default:resnet)')  
    parser.add_argument('--train-method', default='selfgrow', type=str,
                        help='candidates: fixedsplit /selfgrow /autosplit')

    ########################Offline Training#########################
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0)')
    parser.add_argument('--percent', type=float, default=0.1,
                        help='prune rate (default: 0.1)')
    parser.add_argument('--offline-batch-size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=24, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--offline-epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--offline-lr', type=float, default=0.1, metavar='OFFLR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--offline-momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--offline-weight-decay', '--offwd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--early-exit-acc', type=float, default=0.9, 
                        help='expected accuracy of early exit branch (default: 0.9)')
    parser.add_argument('--arch', default='resnet', type=str, 
                        help='architecture to use')
    parser.add_argument('--no-cuda', default=False,  action='store_true',
                        help='cuda use')
    parser.add_argument('--save', default='./result/', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--save_client', default='result/client/', type=str, metavar='PATH',#path change
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--save_server', default='result/server/', type=str, metavar='PATH',#path change
                        help='path to save prune model (default: current directory)')
    

    args = parser.parse_args()

    if not os.path.exists(args.save_client):  # 此处建立也不对 太晚了  此时这个文件夹内已经接收到的模型参数
        os.mkdir(args.save_client)
 
    if torch.cuda.is_available() and not args.no_cuda:
        args.cuda = True
    else:
        args.cuda = False

    net = torch.load(os.path.join(args.save_client, 'model_best.pth.tar')) 
    if args.train_method!='autosplit':
        model_init = model_name[args.model](dataset_class_num[args.dataset],cfg=net['cfg'])      
    else:
        model_init = model_name[args.model](dataset_class_num[args.dataset],split_layer=net['split_layer'],cfg=net['cfg'])    
    model = model_init.NE
    model.load_state_dict(net['NE_state_dict'])
    if args.cuda:
        model.cuda()

    LOGGER = Logger(log_name="client " + str(args.rank))

    trainer = FedAvgClientTrainer(model, cuda=args.cuda)


    dataset = online_dataset_name[args.dataset](root='./dataset/'+args.dataset,
                                path='./dataset/'+args.dataset,
                                dataname=args.dataset,
                                num_clients=args.world_size - 1,
                                seed=args.seed,
                                transform=transforms.ToTensor())

    if args.rank == 1:
        dataset.preprocess()

    trainer.setup_dataset(dataset)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=args.rank,
        ethernet=args.ethernet,
    )

    manager_ = PassiveClientManager(trainer=trainer,
                                    network=network,
                                    logger=LOGGER)
    manager_.run(args)
