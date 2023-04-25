import hydra
import os
import sys
import json
import torch
import pickle
import numpy as np
from typing import Dict
from tqdm import tqdm
from sklearn.svm import SVC, OneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rlmeta.core.types import Action
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict
import rlmeta.utils.nested_utils as nested_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CacheAttackerDetectorEnv, CacheAttackerDetectorEnvFactory
from model import CachePPOTransformerModel
from agent import PPOAgent, SpecAgent, CycloneAgent, PrimeProbeAgent
from utils.trace_parser import load_trace

LABEL={ 'attacker':1,
        'benign':0,
        }

def unbatch_action(action: Action) -> Action:
    act, info = action
    act.squeeze_(0)
    info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
    return Action(act, info)


def run_loop(env: Env, agents, victim_addr=-1) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0
    detector_count = 0.0
    detector_acc = 0.0
    """
    oneclass svm uses fully benign dataset
    """
    env.env.opponent_weights = [1, 0.0]
    if victim_addr == -1:
        timestep = env.reset()
    else:
        timestep = env.reset(victim_address=victim_addr)
    #print("victim address: ", env.env.victim_address )
    for agent_name, agent in agents.items():
        agent.observe_init(timestep[agent_name])
    while not timestep["__all__"].done:
        # Model server requires a batch_dim, so unsqueeze here for local runs.
        actions = {}
        for agent_name, agent in agents.items():
            timestep[agent_name].observation.unsqueeze_(0)
            #print("attacker obs")
            #print(timestep["attacker"].observation)
            action = agent.act(timestep[agent_name])
            # Unbatch the action.
            if isinstance(action, tuple):
                action = Action(action[0], action[1])
            if not isinstance(action.action, (int, np.int64)):
                action = unbatch_action(action)
            actions.update({agent_name:action})
        #print(actions)
        timestep = env.step(actions)

        for agent_name, agent in agents.items():
            agent.observe(actions[agent_name], timestep[agent_name])

        episode_length += 1
        episode_return += timestep['attacker'].reward

        try:
            detector_action = actions['detector'].action.item()
        except:
            detector_action = actions['detector'].action
        if timestep["__all__"].done and detector_action ==1:
            detector_count += 1
        detector_accuracy = detector_count

    metrics = {
        "episode_length": episode_length,
        "episode_return": episode_return,
        "detector_accuracy": detector_accuracy,
    }

    #Cyclone
    return agents['detector'].cyclone_counters, env.env.opponent_agent

def collect(cfg, num_samples):
    # load agents and 
    # run environment loop to collect data
    # return 
    env_fac = CacheAttackerDetectorEnvFactory(cfg.env_config)
    env = env_fac(index=0)

    # Load model
    # Attacker
    if len(cfg.attacker_checkpoint) > 0:
        cfg.model_config["output_dim"] = env.action_space.n
        attacker_params = torch.load(cfg.attacker_checkpoint)
        attacker_model = CachePPOTransformerModel(**cfg.model_config)
        attacker_model.load_state_dict(attacker_params)
        attacker_model.eval()
        attacker_agent = PPOAgent(attacker_model, deterministic_policy=cfg.deterministic_policy) # use ppo agent as attacker
    else:
        attacker_agent = PrimeProbeAgent(cfg.env_config) # use prime+probe agent as attacker
    detector_agent = CycloneAgent(cfg.env_config)
    X, y = [], [] 
    
    for trace_file in cfg.trace_files:
        spec_trace = load_trace(trace_file,
                                limit=cfg.trace_limit,
                                legacy_trace_format=cfg.legacy_trace_format)
        benign_agent = SpecAgent(cfg.env_config, spec_trace, legacy_trace_format=cfg.legacy_trace_format)
        agents = {"attacker": attacker_agent, "detector": detector_agent, "benign": benign_agent}
        for i in tqdm(range(num_samples)):
            x, label = run_loop(env, agents)
            X.append(x)
            y.append(LABEL[label])
    X = np.array(X) #num_samples, m, n = X.shape
    X = X.reshape(num_samples*len(cfg.trace_files), -1)
    y = np.array(y)
    #print('features:\n',X,'\nlabels\n',y)
    return X, y

def train(cfg):
    # run data collection and 
    # train the svm classifier
    # report accuracy
    
    data_file=None

    if data_file is None:
        X_train, y_train = collect(cfg, num_samples=2000)
        X_test, y_test = collect(cfg, num_samples=10)
        data = {"X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test}
        pickle.dump(data, open('data.pkl','wb'))
    else:
        data=pickle.load(open(data_file,'rb'))
        X_train=data['X_train']
        y_train=data['y_train']
        X_test=data['X_test']
        y_test=data['y_test']
    
    
    clf = make_pipeline(
            StandardScaler(),
            OneClassSVM(
                gamma='auto',
                nu=0.01))
    clf.fit(X_train)
    y = clf.predict(X_train)
    train_accuracy = np.mean(y==1)
    y = clf.predict(X_test)
    test_accuracy = np.mean(y==1)

    print("Train Accuracy:",train_accuracy)
    print("Test Accuracy:",test_accuracy)
    
    print("saving the classfier")
    pickle.dump(clf,open('cyclone.pkl','wb'))
    return clf

@hydra.main(config_path="../config", config_name="cyclone")
def main(cfg):
    clf = train(cfg)
    return clf

if __name__=='__main__':
    clf = main()
