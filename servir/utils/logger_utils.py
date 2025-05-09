import os
import time
import logging


from servir.utils.main_utils import collect_env_info, collect_method_info 


def logging_setup(log_path, fname='log.log'):
    """Set up logger to save the running log.
    Args:
        log_path_path (str): Path to save the log.
    """

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    fname_wt = fname.rsplit('.', 1)[0] + '_' + timestamp + '.log'

    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(log_path, fname_wt),
                        filemode='a', format='%(asctime)s - %(message)s', 
                        force=True)
    


def logging_env_info():
    # log env info
    dash_line = '-' * 60 + '\n'

    env_info_dict = collect_env_info()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])

    logging.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)


def logging_config_info(config):
    # log config
    if config is not None:
        dash_line = '-' * 60 + '\n'
        config_info = '\n'.join([(f'{k}: {v}') for k, v in config.items()])
        logging.info('Config Info:\n' + dash_line + config_info + '\n' + dash_line)

def logging_method_info(config, method, device):

    info, flops = collect_method_info(config, method, device)
    """log the basic infomation of supported methods"""
    dash_line = '-' * 80 + '\n'

    logging.info('Model info:\n' + info+'\n' + flops+'\n' + dash_line)
