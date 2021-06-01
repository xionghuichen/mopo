import argparse
from ppo import PPO

def arg_parse():
    parser = argparse.ArgumentParser(description='adaptable test')
    parser.add_argument('--adapt', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--context', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    print('use rnn: {}\ntest mode: {}\nuse context: {}'.format(
        args.adapt, args.test, args.context
    ))
    ppo = PPO(not args.adapt, args.context)
    if args.test:
        ppo.test()
    else:
        ppo.run()