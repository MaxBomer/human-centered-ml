#!/usr/bin/env python3
"""Quick test script for fast validation of active learning algorithms."""
from __future__ import annotations

import argparse
import numpy as np
import torch
import warnings
import os
import sys

import arguments
from parameters import *
from utils import *

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def quick_test(
    strategy_name: str = 'RandomSampling',
    dataset_name: str = 'MNIST',
    init_seed: int = 100,
    quota: int = 200,
    batch_size: int = 50,
    seed: int = 4666,
    gpu: int = 0
) -> None:
    """Run a quick test with minimal iterations."""
    
    print(f"Quick Test: {strategy_name} on {dataset_name}")
    print(f"Initial labeled: {init_seed}, Quota: {quota}, Batch: {batch_size}")
    print("-" * 60)
    
    # Set up environment
    os.environ['TORCH_HOME'] = './basicmodel'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    # Fix random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    
    # Get data, network, strategy
    args_task = args_pool[dataset_name]
    dataset = get_dataset(dataset_name, args_task)
    
    if strategy_name == 'LossPredictionLoss':
        net = get_net_lpl(dataset_name, args_task, device)
    elif strategy_name == 'WAAL':
        net = get_net_waal(dataset_name, args_task, device)
    else:
        net = get_net(dataset_name, args_task, device)
    
    strategy = get_strategy(strategy_name, dataset, net, None, args_task)
    
    # Initialize labels
    dataset.initialize_labels(init_seed)
    num_rounds = int(quota / batch_size)
    
    print(f"Starting with {init_seed} labeled samples")
    print(f"Will run {num_rounds} rounds of {batch_size} samples each\n")
    
    # Train initial model
    if strategy_name == 'WAAL':
        strategy.train(model_name=strategy_name)
    else:
        strategy.train()
    
    # Evaluate initial accuracy
    preds = strategy.predict(dataset.get_test_data())
    initial_acc = dataset.cal_test_acc(preds)
    print(f"Round 0: Accuracy = {initial_acc:.4f}")
    
    # Run active learning rounds
    for rd in range(1, num_rounds + 1):
        print(f"Round {rd}: Querying {batch_size} samples...", end=" ")
        
        # Query
        q_idxs = strategy.query(batch_size)
        
        # Update
        strategy.update(q_idxs)
        
        # Train
        if strategy_name == 'WAAL':
            strategy.train(model_name=strategy_name)
        else:
            strategy.train()
        
        # Evaluate
        preds = strategy.predict(dataset.get_test_data())
        acc = dataset.cal_test_acc(preds)
        print(f"Accuracy = {acc:.4f} (labeled: {init_seed + rd * batch_size})")
    
    print("\n" + "=" * 60)
    print(f"Final accuracy: {acc:.4f}")
    print(f"Total labeled samples: {init_seed + num_rounds * batch_size}")
    print("Quick test completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick test for active learning')
    parser.add_argument('-a', '--ALstrategy', default='RandomSampling', type=str,
                       help='Active learning strategy')
    parser.add_argument('-d', '--dataset_name', default='MNIST', type=str,
                       help='Dataset name')
    parser.add_argument('-s', '--initseed', default=100, type=int,
                       help='Initial labeled samples')
    parser.add_argument('-q', '--quota', default=200, type=int,
                       help='Total quota')
    parser.add_argument('-b', '--batch', default=50, type=int,
                       help='Batch size per round')
    parser.add_argument('--seed', default=4666, type=int,
                       help='Random seed')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                       help='GPU ID')
    
    args = parser.parse_args()
    
    quick_test(
        strategy_name=args.ALstrategy,
        dataset_name=args.dataset_name,
        init_seed=args.initseed,
        quota=args.quota,
        batch_size=args.batch,
        seed=args.seed,
        gpu=args.gpu
    )

