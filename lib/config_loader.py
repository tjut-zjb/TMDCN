import argparse
import configparser
import os


def setTerminal():
    """
    Set command line parameters and parse.
    Returns a namespace object containing command-line parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        default='PEMS04',
                        type=str,
                        help="select dataset(default: PEMS04)")
    args = parser.parse_args()

    return args


def getConfig(args):
    """
    Read configuration files and return data and train configurations.
    """
    config = configparser.ConfigParser()
    config_path = os.path.join('./configuration', f'{args.dataset}.conf')
    config.read(config_path)

    data_config = config['Data']
    train_config = config['Train']

    return data_config, train_config
