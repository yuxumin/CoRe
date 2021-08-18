from tools import run_net
from tools import test_net
from utils import parser

def main():
    # config
    args = parser.get_args()
    parser.setup(args)   
    if args.benchmark == 'MTL':
        if not args.usingDD:
            args.score_range = 100
    print(args)
    # run
    if args.test:
        test_net(args)
    else:
        run_net(args)


if __name__ == '__main__':
    main()