# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# @Update : 13/09/2022, Ruihong Qiu

"""
recbole.quick_start
########################
"""

import tqdm

import os.path

import numpy as np
import _pickle as pickle

from sklearn.svm import SVC
# from thundersvm import SVC
from sklearn.metrics.pairwise import cosine_similarity

from RNTK_avg import RNTK_first_time_step, RNTK_lin, RNTK_relu

from recbole.utils import ensure_dir
from recbole.evaluator import Evaluator, Collector

import logging
from logging import getLogger

import torch
import pickle

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    # init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    
    logger.info(config)
    
    # dataset filtering
    dataset = create_dataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)
    
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    if config['save_dataloaders']:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))
    
    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    
    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)
    
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file, dataset_file=None, dataloader_file=None):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.
        dataset_file (str, optional): The path of filtered dataset. Defaults to ``None``.
        dataloader_file (str, optional): The path of split dataloaders. Defaults to ``None``.

    Note:
        The :attr:`dataset` will be loaded or created according to the following strategy:
        If :attr:`dataset_file` is not ``None``, the :attr:`dataset` will be loaded from :attr:`dataset_file`.
        If :attr:`dataset_file` is ``None`` and :attr:`dataloader_file` is ``None``,
        the :attr:`dataset` will be created according to :attr:`config`.
        If :attr:`dataset_file` is ``None`` and :attr:`dataloader_file` is not ``None``,
        the :attr:`dataset` will neither be loaded or created.

        The :attr:`dataloader` will be loaded or created according to the following strategy:
        If :attr:`dataloader_file` is not ``None``, the :attr:`dataloader` will be loaded from :attr:`dataloader_file`.
        If :attr:`dataloader_file` is ``None``, the :attr:`dataloader` will be created according to :attr:`config`.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_logger(config)
    
    dataset = None
    if dataset_file:
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
    
    if dataloader_file:
        train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)
    else:
        if dataset is None:
            dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
    
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    
    return config, model, dataset, train_data, valid_data, test_data


def run_kernel_calc_pytorch_logic(model=None, dataset=None, config_file_list=None, config_dict=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    
    logger.info(config)
    
    # dataset filtering
    dataset = create_dataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)
    
    # logging setting
    LOGROOT = config['log_root'] if config['log_root'] else './log/'
    LOGROOT += config['dataset'] + '/OverRec-aug-{}/w{}_u{}_b{}_v{}_{}_len{}_{}{}/'.format(
        config['aug'], config['varw'], config['varu'], config['varb'], config['varv'], config['phi'],
        config['MAX_ITEM_LIST_LENGTH'], config['user_inter_num_interval'][1:-1],
        config['item_inter_num_interval'][1:-1])
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)
    
    config['log_dir'] = LOGROOT
    logfilename = 'log.txt'
    
    logfilepath = os.path.join(LOGROOT, logfilename)
    
    # build and split datasets (padding)
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets
    
    assert (train_dataset.item_num == valid_dataset.item_num) and (train_dataset.item_num == test_dataset.item_num)
    
    logging.warning('kernel mode-{}'.format(config['kernel_mode']))
    
    logging.warning('stepwise logical dot product and calculate kernel')
    
    # rntk calculation
    param = {'sigmaw': config['varw'], 'sigmau': config['varu'], 'sigmab': config['varb'], 'sigmah': config['varh'],
             'sigmav': config['varv']}
    
    if config['phi'] == 'lin':
        RNTK_non = RNTK_lin
    elif config['phi'] == 'relu':
        RNTK_non = RNTK_relu
    
    for i in tqdm.tqdm(range(config['MAX_ITEM_LIST_LENGTH'])):
        if config['kernel_mode'] == 1:
            ver = train_dataset['item_id_list'][:, i].unsqueeze(dim=1).repeat(
                [1, train_dataset['item_id_list'].shape[0]])
            hor = ver
        elif config['kernel_mode'] == 2:
            ver = valid_dataset['item_id_list'][:, i].unsqueeze(dim=1).repeat(
                [1, train_dataset['item_id_list'].shape[0]])
            hor = train_dataset['item_id_list'][:, i].unsqueeze(dim=0).repeat(
                [valid_dataset['item_id_list'].shape[0], 1])
        elif config['kernel_mode'] == 3:
            ver = test_dataset['item_id_list'][:, i].unsqueeze(dim=1).repeat(
                [1, train_dataset['item_id_list'].shape[0]])
            hor = train_dataset['item_id_list'][:, i].unsqueeze(dim=0).repeat(
                [test_dataset['item_id_list'].shape[0], 1])
        
        # dot = torch.logical_and(ver == hor, ver > 0)
        
        logging.warning(i)
        if i == 0:
            rntk, nngp = RNTK_first_time_step(ver, hor, param)
        else:
            rntk, nngp = RNTK_non(ver, hor, rntk, torch.clamp(nngp, -1, 1), param, vt_mask, False)
        vt_mask = torch.logical_and(hor > 0, ver > 0)
    
    rntk, nngp = RNTK_non(None, None, rntk, torch.clamp(nngp, -1, 1), param, vt_mask, True)
    for k in [1, 5, 10]:
        config['knn'] = k
        weighted_y_eval(rntk, train_dataset, test_dataset, config, LOGROOT, model, kernel='rntk')
        weighted_y_eval(nngp, train_dataset, test_dataset, config, LOGROOT, model, kernel='nngp')
    
    # calculate RNTK with precomputed dot product
    # logging.warning('saving')
    # np.save(os.path.join(LOGROOT, 'rntk-{}.npy'.format(config['kernel_mode'])), rntk.numpy())
    # np.save(os.path.join(LOGROOT, 'nngp-{}.npy'.format(config['kernel_mode'])), nngp.numpy())


def run_sknn(model=None, dataset=None, config_file_list=None, config_dict=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    config['aug'] = 'no'
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    
    logger.info(config)
    
    # dataset filtering
    dataset = create_dataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)
    
    # logging setting
    LOGROOT = config['log_root'] if config['log_root'] else './log/'
    LOGROOT += config['dataset'] + '/{}-aug-{}/'.format(config['model'],
                                                        config['aug'])
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)
    
    config['log_dir'] = LOGROOT
    logfilename = '{}-log.txt'.format(config['knn'])
    
    logfilepath = os.path.join(LOGROOT, logfilename)
    
    # build and split datasets (padding)
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets
    
    assert (train_dataset.item_num == valid_dataset.item_num) and (train_dataset.item_num == test_dataset.item_num)
    
    logging.warning('kernel mode-{}'.format(config['kernel_mode']))
    
    logging.warning('stepwise logical dot product and calculate kernel')
    
    # sequence similarity calculation
    # training sequences: [num_train_seq, num_all_item]
    # test sequences: [num_train_seq, num_all_item]
    train_mat = np.zeros([train_dataset['item_id_list'].shape[0], train_dataset.item_num - 1])
    test_mat = np.zeros([test_dataset['item_id_list'].shape[0], test_dataset.item_num - 1])
    
    for i in tqdm.tqdm(range(train_dataset['item_id_list'].shape[0])):
        # normal training input items. Minus 1 for padding
        item_ids = train_dataset['item_id_list'][i][:train_dataset['item_length'][i]] - 1
        train_mat[i, item_ids] = 1.
        
        # normal training target item.
        train_mat[i, train_dataset['item_id'][i] - 1] = 1.
        
        # testing input items.
        item_ids = test_dataset['item_id_list'][i][:test_dataset['item_length'][i]] - 1
        test_mat[i, item_ids] = 1.
    
    # SKNN is originally for session recommendation.
    # Different testing strategy: session-based test on new sessions.
    # sequential recommendation test on the same sequence but new interaction
    # So still using training dataset for similarity calculation but setting the input sequence as 0.
    cos_sim = cosine_similarity(test_mat, train_mat)
    cos_sim = cos_sim * (1 - np.eye(cos_sim.shape[0]))
    
    ind_topknn = np.argpartition(cos_sim, -config['knn'], axis=-1)[:, -config['knn']:]
    # ind_10 = np.argpartition(cos_sim, -10, axis=-1)[:, -10:]
    # ind_20 = np.argpartition(cos_sim, -20, axis=-1)[:, -20:]
    
    test_scores = np.zeros([test_dataset['item_id_list'].shape[0], test_dataset.item_num - 1])
    for i in tqdm.tqdm(range(test_dataset['item_id_list'].shape[0])):
        ind = ind_topknn[i]
        sim_of_nei = cos_sim[i][ind]
        scores = (sim_of_nei[:, None] * train_mat[ind]).sum(axis=0)[None, :]
        test_scores[i] = scores
    
    # eval
    logfile = open(logfilepath, 'w')
    eval_collector = Collector(config)
    eval_collector.eval_batch_collect(torch.tensor(test_scores), None,
                                      torch.tensor(range(test_scores.shape[0])),
                                      test_dataset['item_id'] - 1)
    
    eval_collector.model_collect(model)
    struct = eval_collector.get_data_struct()
    
    evaluator = Evaluator(config)
    result = evaluator.evaluate(struct)
    
    logging.warning(result)
    logfile.write(str(result) + '\n')


def run_stan(model=None, dataset=None, config_file_list=None, config_dict=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    config['aug'] = 'no'
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    
    logger.info(config)
    
    # dataset filtering
    dataset = create_dataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)
    
    # logging setting
    LOGROOT = config['log_root'] if config['log_root'] else './log/'
    LOGROOT += config['dataset'] + '/{}-aug-{}/'.format(config['model'],
                                                        config['aug'])
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)
    
    config['log_dir'] = LOGROOT
    logfilename = '{}-lmd1-{}-lmd3-{}-log.txt'.format(config['knn'], config['lambda1'], config['lambda3'])
    
    logfilepath = os.path.join(LOGROOT, logfilename)
    
    # build and split datasets (padding)
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets
    
    assert (train_dataset.item_num == valid_dataset.item_num) and (train_dataset.item_num == test_dataset.item_num)
    
    logging.warning('kernel mode-{}'.format(config['kernel_mode']))
    
    logging.warning('stepwise logical dot product and calculate kernel')
    
    # sequence similarity calculation with \lambda_1 and position for testing sequences according to the paper
    # this weighting is not for training sequences
    # training sequences: [num_train_seq, num_all_item]
    # test sequences: [num_train_seq, num_all_item]
    train_mat = np.zeros([train_dataset['item_id_list'].shape[0], train_dataset.item_num - 1])
    test_mat = np.zeros([test_dataset['item_id_list'].shape[0], test_dataset.item_num - 1])
    
    for i in tqdm.tqdm(range(train_dataset['item_id_list'].shape[0])):
        # normal training input items. Minus 1 for padding
        item_ids = train_dataset['item_id_list'][i][:train_dataset['item_length'][i]] - 1
        train_mat[i, item_ids] = 1.
        
        # normal training target item.
        train_mat[i, train_dataset['item_id'][i] - 1] = 1.
        
        # testing input items.
        item_ids = test_dataset['item_id_list'][i][:test_dataset['item_length'][i]] - 1
        weights = np.exp(
            np.arange(test_dataset['item_length'][i].numpy()) - (test_dataset['item_length'][i].numpy() - 1) / config[
                'lambda1'])
        test_mat[i, item_ids] = weights
    
    # STAN is originally for session recommendation.
    # Different testing strategy: session-based test on new sessions.
    # sequential recommendation test on the same sequence but new interaction
    # So still using training dataset for similarity calculation but setting the input sequence as 0.
    cos_sim = cosine_similarity(test_mat, train_mat)
    cos_sim = cos_sim * (1 - np.eye(cos_sim.shape[0]))
    
    kernel_stan_eval(cos_sim, train_dataset, test_dataset, config, LOGROOT, model, kernel='stan')
    stan_eval(cos_sim, train_dataset, test_dataset, config, LOGROOT, model, kernel='stan')
    weighted_y_eval(cos_sim, train_dataset, test_dataset, config, LOGROOT, model, kernel='stan')


def stan_eval(sim_mat, train_dataset, test_dataset, config, LOGROOT, model, kernel):
    ind_topknn = np.argpartition(sim_mat, -config['knn'], axis=-1)[:, -config['knn']:]
    test_scores = np.zeros([test_dataset['item_id_list'].shape[0], test_dataset.item_num - 1])
    for i in tqdm.tqdm(range(test_dataset['item_id_list'].shape[0])):
        ind = ind_topknn[i]
        
        test_seq = test_dataset['item_id_list'][i][:test_dataset['item_length'][i]]
        
        scores_mat = np.zeros([config['knn'], train_dataset.item_num - 1])
        for k, seq_id in enumerate(ind):
            train_seq = train_dataset['item_id_list'][seq_id][:train_dataset['item_length'][seq_id]]
            
            # find the latest common_items in training session
            common_items_idx = -1
            for j, item in enumerate(train_seq):
                if item in test_seq:
                    common_items_idx = j
            
            if common_items_idx == -1:
                print("no common item")
                continue
            
            item_scores = np.exp(
                -np.abs(np.arange(train_dataset['item_length'][seq_id]) - common_items_idx) / config['lambda3'])
            scores_mat[k, train_seq - 1] = item_scores
        
        scores = scores_mat.sum(axis=0)[None, :]
        test_scores[i] = scores
    
    if kernel == 'rntk':
        logfilename = 'stan-rntk-{}-log.txt'.format(config['knn'], config['lambda3'])
    elif kernel == 'nngp':
        logfilename = 'stan-nngp-{}-log.txt'.format(config['knn'], config['lambda3'])
    elif kernel == 'stan':
        logfilename = '{}-log.txt'.format(config['knn'], config['lambda3'])
    logfilepath = os.path.join(LOGROOT, logfilename)
    
    eval(test_dataset, test_scores, config, model, logfilepath)


def weighted_y_eval(sim_mat, train_dataset, test_dataset, config, LOGROOT, model, kernel):
    ind_topknn = np.argpartition(sim_mat, -config['knn'], axis=-1)[:, -config['knn']:]
    test_scores = np.zeros([test_dataset['item_id_list'].shape[0], test_dataset.item_num - 1])
    
    for i in tqdm.tqdm(range(test_dataset['item_id_list'].shape[0])):
        ind = ind_topknn[i]
        
        sim_seq_targets = train_dataset['item_id'][ind] - 1
        seq_sim = sim_mat[i][ind]
        
        scores = np.zeros([1, train_dataset.item_num - 1])
        scores[0, sim_seq_targets] = seq_sim
        test_scores[i] = scores
    
    if kernel == 'rntk':
        logfilename = 'weighted-y-rntk-{}-log.txt'.format(config['knn'], config['lambda3'])
    elif kernel == 'nngp':
        logfilename = 'weighted-y-nngp-{}-log.txt'.format(config['knn'], config['lambda3'])
    elif kernel == 'stan':
        logfilename = 'weighted-y-stan-{}-log.txt'.format(config['knn'], config['lambda3'])
    logfilepath = os.path.join(LOGROOT, logfilename)
    
    eval(test_dataset, test_scores, config, model, logfilepath)


def kernel_stan_eval(sim_mat, train_dataset, test_dataset, config, LOGROOT, model, kernel):
    ind_topknn = np.argpartition(sim_mat, -config['knn'], axis=-1)[:, -config['knn']:]
    test_scores = np.zeros([test_dataset['item_id_list'].shape[0], test_dataset.item_num - 1])
    for i in tqdm.tqdm(range(test_dataset['item_id_list'].shape[0])):
        ind = ind_topknn[i]
        
        test_seq = test_dataset['item_id_list'][i][:test_dataset['item_length'][i]]
        
        scores_mat = np.zeros([config['knn'], train_dataset.item_num - 1])
        for k, seq_id in enumerate(ind):
            train_seq = torch.cat([train_dataset['item_id_list'][seq_id][:train_dataset['item_length'][seq_id]],
                                   train_dataset['item_id'][seq_id].unsqueeze(0)])
            
            # find the latest common_items in training session
            common_items_idx = -1
            for j, item in enumerate(train_seq):
                if item in test_seq:
                    common_items_idx = j
            
            if common_items_idx == -1:
                print("no common item")
                continue
            
            # +1 for target item position and -1 in position compensates for new interaction
            item_scores = np.exp(
                -np.abs(np.arange(train_dataset['item_length'][seq_id] + 1) - common_items_idx - 1) / config['lambda3'])
            scores_mat[k, train_seq - 1] = item_scores
        
        scores = scores_mat.sum(axis=0)[None, :]
        test_scores[i] = scores
    
    if kernel == 'rntk':
        logfilename = 'kernel-stan-rntk-{}-lmd3-{}-log.txt'.format(config['knn'], config['lambda3'])
    elif kernel == 'nngp':
        logfilename = 'kernel-stan-nngp-{}-lmd3-{}-log.txt'.format(config['knn'], config['lambda3'])
    elif kernel == 'stan':
        logfilename = 'kernel-stan-{}-lmd3-{}-log.txt'.format(config['knn'], config['lambda3'])
    logfilepath = os.path.join(LOGROOT, logfilename)
    
    eval(test_dataset, test_scores, config, model, logfilepath)


def eval(test_dataset, test_scores, config, model, logfilepath):
    logfile = open(logfilepath, 'w')
    eval_collector = Collector(config)
    eval_collector.eval_batch_collect(torch.tensor(test_scores), None,
                                      torch.tensor(range(test_scores.shape[0])),
                                      test_dataset['item_id'] - 1)
    
    eval_collector.model_collect(model)
    struct = eval_collector.get_data_struct()
    
    evaluator = Evaluator(config)
    result = evaluator.evaluate(struct)
    
    logging.warning(result)
    logfile.write(str(result) + '\n')
