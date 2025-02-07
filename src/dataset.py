import ogbench
import gymnasium
import numpy as np 
import collections
import datetime
import io
import pathlib
import uuid
import os

def load_dataset(dataset_path, directory, ob_dtype=np.float32, action_dtype=np.float32, compact_dataset=False):
    
    # file = np.load(dataset_path)
    with np.load(dataset_path, mmap_mode='r') as file:
        dataset = dict()
        for k in ['observations', 'actions', 'terminals']:
            if k == 'observations':
                dtype = ob_dtype
            elif k == 'actions':
                dtype = action_dtype
            else:
                dtype = np.float32
            dataset[k] = file[k][...].astype(dtype, copy=False)
    del file
    pos = np.where(dataset['terminals'] == 1)[0]+1
    start = 0 
    trajectory = 0
    # if visual --> uint8 check if its 255 ot 0 to 1 
    for i in pos:
        episode_length = len(dataset['terminals'][start:i])
        temp = {
            'observation' : np.transpose(dataset['observations'][start:i], (0, 3, 1, 2)), # channel first
            'is_first': np.zeros(len(dataset['terminals'][start:i])).astype(bool),
            'is_last': dataset['terminals'][start:i].astype(bool),
            'is_terminal': dataset['terminals'][start:i].astype(bool),
            'action': dataset['actions'][start:i], 
            'reward': ((np.arange(episode_length) + 1) / episode_length).astype(np.float32),
            'discount': np.expand_dims(1 - dataset['terminals'][start:i], axis=1).astype(np.float32),
        }
        if temp['is_terminal'].sum() != 1:
            print('ERROR')
        else:
            trajectory += 1
        save_episode(directory, temp, trajectory, len(temp['is_terminal']))
        # break
        start = i
    print(f"Number of trajectory : {trajectory}")
    return dataset

def save_episode(directory, episode, episode_id, episode_len):
    idx = episode_id
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = episode_len
    filename = pathlib.Path(directory).expanduser() / f'{idx}-{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename

def make_env(dataset_name, **env_kwargs):
    # Make environment.
    # Simplest form 
    splits = dataset_name.split('-')
    env_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the dataset type.
    env = gymnasium.make(env_name, **env_kwargs)
    return env 

def test_make_env(dataset_name):
        train_env = make_env(dataset_name)
        out, goal = train_env.reset()

        for i in range(100):
            action = train_env.action_space.sample()
            obs, reward, terminate, truncate, info = train_env.step(action)
        print('done')

if __name__ == "__main__":
    
    # Make an environment and load datasets.
    dataset_name = 'visual-antmaze-medium-navigate-v0'
    # dataset_name = 'antmaze-large-navigate-v0' 


    dataset = load_dataset(dataset_path='/home/wtc/sai/genrl/data/ogbench/visual-antmaze-medium-navigate-v0-val.npz', 
                        directory='/home/wtc/sai/genrl/data/ogbench/visual-antmaze-medium-navigate',
                        ob_dtype=np.uint8)
    

