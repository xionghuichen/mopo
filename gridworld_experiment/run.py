import argparse
from ppo import PPO

def arg_parse():
    parser = argparse.ArgumentParser(description='adaptable test')
    parser.add_argument('--adapt', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--mujoco', action='store_true')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    print('use rnn: {}\ntest mode: {}\nuse context: {}\nseed: {}'.format(
        args.adapt, args.test, args.context, args.seed
    ))
    ppo = PPO(not args.adapt, args.context, not args.mujoco, args.seed)
    if args.test:
        ppo.test()
    else:
        ppo.run()