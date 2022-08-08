import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(save_dir='saved/log', log_config='logger/logger_config.yml', default_level=logging.INFO):
    '''Setup logging configuration'''
    log_config_path = Path(log_config)
    if log_config_path.is_file():
        with open(log_config, 'r') as f:
            config = yaml.safe_load(f.read())
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = "/".join([save_dir,handler['filename']])

            logging.config.dictConfig(config)
    else:
        print(f"Warning: logging configuration file is not found in {log_config}")
        logging.basicConfig(level=default_level)
