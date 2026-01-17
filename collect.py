import os
from datetime import datetime
import yaml
import multiprocessing
import argparse

from env import SAMPLE_ENVIRONMENT, make_env, Darkroom, DarkroomPermuted, DarkKeyToDoor
from algorithm import ALGORITHM, HistoryLoggerCallback
import h5py
import numpy as np
import torch

from stable_baselines3.common.vec_env import DummyVecEnv

from utils import get_config, get_traj_file_name



def worker(arg, config, traj_dir, env_idx, history, file_name):
    # limit CPU threads in worker to avoid oversubscription
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    
    if config['env'] == 'darkroom':
        env = DummyVecEnv([make_env(config, goal=arg)] * config['n_stream'])
    elif config['env'] == 'dark_key_to_door':
        # arg is (key_x, key_y, goal_x, goal_y)
        key = arg[:2]
        goal = arg[2:]
        env = DummyVecEnv([make_env(config, key=key, goal=goal)] * config['n_stream'])
    else:
        raise ValueError(f'Invalid environment: {config["env"]}')
    
    alg_name = config['alg']
    seed = config['alg_seed'] + env_idx

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    alg = ALGORITHM[alg_name](config, env, seed, traj_dir)
    callback = HistoryLoggerCallback(config['env'], env_idx, history)
    log_name = f'{file_name}_{env_idx}'
    
    alg.learn(total_timesteps=config['total_source_timesteps'],
              callback=callback,
              log_interval=1,
              tb_log_name=log_name,
              reset_num_timesteps=True,
              progress_bar=True)
    env.close()



if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # start method already set (possible on some platforms/runs)
        pass
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='darkroom',
                       help='Environment name: darkroom or dark_key_to_door')
    args = parser.parse_args()
    
    # Determine config files based on environment
    env_config_map = {
        'darkroom': ('darkroom', 'ppo_darkroom'),
        'dark_key_to_door': ('dark_key_to_door', 'ppo_dark_key_to_door'),
    }
    if args.env not in env_config_map:
        raise ValueError(f'Unknown environment: {args.env}')
    env_cfg, alg_cfg = env_config_map[args.env]
    
    config = get_config(f"config/env/{env_cfg}.yaml")
    config.update(get_config(f"config/algorithm/{alg_cfg}.yaml"))

    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)
        
    traj_dir = 'datasets'

    train_args, test_args = SAMPLE_ENVIRONMENT[config['env']](config, shuffle=False)
    total_args = train_args + test_args
    n_envs = len(total_args)

    file_name = get_traj_file_name(config)
    path = f'datasets/{file_name}.hdf5'
    
    start_time = datetime.now()
    print(f'Training started at {start_time}')

    # Use a Manager for shared history; ensure clean shutdown on Ctrl+C
    with multiprocessing.Manager() as manager:
        history = manager.dict()

        pool = None
        try:
            pool = multiprocessing.Pool(processes=config['n_process'])

            # Prepare arguments once to avoid lambda capture issues
            tasks = [(total_args[i], config, traj_dir, i, history, file_name) for i in range(n_envs)]

            # Run workers; this will block until completion or until interrupted
            pool.starmap(worker, tasks)

            # close normally
            pool.close()
            pool.join()

        except KeyboardInterrupt:
            print('KeyboardInterrupt received — terminating workers...')
            if pool is not None:
                pool.terminate()
                pool.join()
        finally:
            # Ensure pool is cleaned up if something else went wrong
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    pass

        # Save whatever history was collected so far (guard missing entries)
        try:
            with h5py.File(path, 'w-') as f:
                for i in range(n_envs):
                    if str(i) in history:
                        env_data = history[str(i)] if isinstance(history[str(i)], dict) else history[i]
                    elif i in history:
                        env_data = history[i]
                    else:
                        # no data collected for this env index
                        continue

                    env_group = f.create_group(f'{i}')
                    for key, value in env_data.items():
                        env_group.create_dataset(key, data=value)
        except Exception as e:
            print(f'Warning: failed to write history to {path}: {e}')

    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    