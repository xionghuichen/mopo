from .logger_base import LoggerBase, get_base_path, system
import os
import numpy as np
import time
import copy


class Logger(LoggerBase):
    def __init__(self, log_to_file=True, force_backup=False, short_name='logging'):
        self.output_dir = os.path.join(get_base_path(), 'log', short_name)
        if log_to_file:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if os.path.exists(os.path.join(self.output_dir, 'log.txt')):
                system(f'mv {os.path.join(self.output_dir, "log.txt")} {os.path.join(self.output_dir, "log_back.txt")}')
            self.log_file = open(os.path.join(self.output_dir, 'log.txt'), 'w')
        else:
            self.log_file = None
            # super(Logger, self).set_log_file(self.log_file)
        super(Logger, self).__init__(self.output_dir, log_file=self.log_file)
        # self.parameter.set_log_func(lambda x: self.log(x))
        self.current_data = {}
        self.logged_data = set()

        self.model_output_dir = self.output_dir
        self.log(f"my output path is {self.output_dir}")

        self.init_tb()
        # self.output_dir = os.path.join(get_base_path(), "log_file")

    @staticmethod
    def get_model_output_path(parameter):
        output_dir = os.path.join(get_base_path(), 'log_file', parameter.short_name)
        return os.path.join(output_dir, 'model')

    @staticmethod
    def get_replay_buffer_path(parameter):
        output_dir = os.path.join(get_base_path(), 'log_file', parameter.short_name)
        return os.path.join(output_dir, 'replay_buffer.pkl')

    def log(self, *args, color=None, bold=True):
        super(Logger, self).log(*args, color=color, bold=bold)

    def log_dict(self, color=None, bold=False, **kwargs):
        for k, v in kwargs.items():
            super(Logger, self).log('{}: {}'.format(k, v), color=color, bold=bold)

    def log_dict_single(self, data, color=None, bold=False):
        for k, v in data.items():
            super(Logger, self).log('{}: {}'.format(k, v), color=color, bold=bold)

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def log_tabular(self, key, val=None, tb_prefix=None, with_min_and_max=False, average_only=False, no_tb=False):
        if val is not None:
            super(Logger, self).log_tabular(key, val, tb_prefix, no_tb=no_tb)
        else:
            if key in self.current_data:
                self.logged_data.add(key)
                super(Logger, self).log_tabular(key if average_only else "Average"+key, np.mean(self.current_data[key]), tb_prefix, no_tb=no_tb)
                if not average_only:
                    super(Logger, self).log_tabular("Std" + key,
                                                    np.std(self.current_data[key]), tb_prefix, no_tb=no_tb)
                    if with_min_and_max:
                        super(Logger, self).log_tabular("Min" + key, np.min(self.current_data[key]), tb_prefix, no_tb=no_tb)
                        super(Logger, self).log_tabular('Max' + key, np.max(self.current_data[key]), tb_prefix, no_tb=no_tb)

    def add_tabular_data(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.current_data:
                self.current_data[k] = []
            if not isinstance(v, list):
                self.current_data[k].append(v)
            else:
                self.current_data[k] += v
    
    def dump_tabular(self):
        for k in self.current_data:
            if k not in self.logged_data:
                self.log_tabular(k, average_only=True)
        self.logged_data.clear()
        self.current_data.clear()
        super(Logger, self).dump_tabular()

def main():
    pass

if __name__ == '__main__':
    logger = Logger(True, False, 'logging_test')
    for i in range(10):
        logger.log_tabular('a', i)
        logger.dump_tabular()

