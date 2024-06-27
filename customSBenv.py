import gymnasium as gym
import numpy as np
from gymnasium import spaces
import simfire
from simfire.enums import GameStatus, BurnStatus
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from numpy import zeros,newaxis
import time
import random
import os
from datetime import datetime
import math
import copy

def save_array_to_file(array, filename):
    array_str = '|'.join(','.join(str(cell) for cell in row) for row in array)
    with open(filename, 'w') as file:
        file.write(array_str)
    

def get_burning(mp):
    count = 0
    for row in mp:
        for item in row:
            if item == 1:
                count +=1
    return count

def get_burned(mp):
    count = 0
    for row in mp:
        for item in row:
            if item == 2:
                count +=1
    return count

def get_unburned(mp):
    count = 0
    for row in mp:
        for item in row:
            if item == 0:
                count +=1
    return count   
 
def get_mitigated(mp):
    count = 0
    for row in mp:
        for item in row:
            if item == 3:
                count +=1
    return count 

def get_total(mp):
    burning = get_burning(mp)
    burned = get_burned(mp)
    unburned = get_unburned(mp)
    return burned+burning+unburned

def get_reward(mp):
    burning = get_burning(mp)
    burned = get_burned(mp)
    unburned = get_unburned(mp)

    return -10*(burning+burned)/(burning+burned+unburned)

def get_reward_l2(mp,pmp, x, y, target="fire"):
    if target == "fire":
        dist, square = distance_to_fire(mp,x,y)
    if target == "prob":
        dist, square = distance_to_prob(pmp, x, y, 0.5)

    return -10*(get_burning(mp)+get_burned(mp))/get_total(mp) - 0.5*dist

def get_reward_l2_vel(self, target="fire", atarget = "fire"):
    if target == "fire":
        dist, square = distance_to_fire(self.fire_map,self.agent_x,self.agent_y)
    if target == "prob":
        dist, square = distance_to_prob(self.prob_map, self.agent_x, self.agent_y, 0.5)
    if atarget == "fire":
        p1 = (get_burned(self.prev_map3)+get_burning(self.prev_map3))
        p2 = (get_burned(self.prev_map2)+get_burning(self.prev_map2))
        p3 = (get_burned(self.prev_map)+get_burning(self.prev_map))
        p4 = (get_burned(self.fire_map)+get_burning(self.fire_map))

        v1 = (p2 - p1)/p1
        v2 = (p3 - p2)/p2
        v3 = (p4 - p3)/p3

        a1 = v2 - v1
        a2 = v3 - v2
    elif atarget == "prob":
        p1 = np.sum(self.prev_prob3)
        p2 = np.sum(self.prev_prob2)
        p3 = np.sum(self.prev_prob)
        p4 = np.sum(self.prob_map)

        v1 = (p2 - p1)/p1
        v2 = (p3 - p2)/p2
        v3 = (p4 - p3)/p3

        if p1 == 0:
            v1 = 0
        if p2 == 0:
            v2 = 0
        if p3 == 0:
            v3 = 0

        a1 = v2 - v1
        a2 = v3 - v2

    return -10*(get_burning(self.fire_map)+get_burned(self.fire_map))/get_total(self.fire_map) - 2*dist - 10*(v3)
    
def get_reward_bench(mp, pmp, step,target="fire"):
    mp_total = get_burned(mp)+get_burning(mp)+get_mitigated(mp)
    bmp = np.load("benchmarks//"+str(step)+".npy")
    bmp_total =  get_burned(bmp)+get_burning(bmp)
    emp = np.load("benchmarks//605.npy")
    emp_total =  get_burned(emp)+get_burning(emp)

    pmp_total = np.sum(pmp)
    bpmp = generate_probs_from_bench(step)
    bpmp_total = np.sum(bpmp)
    return (bmp_total-mp_total)/emp_total


def run_one_simulation_step(self, total_updates):
    num_updates = 0

    while self.sim.fire_status == GameStatus.RUNNING and num_updates < total_updates:
        self.sim.fire_sprites = self.sim.fire_manager.sprites
        self.sim.fire_map, self.sim.fire_status = self.sim.fire_manager.update(self.sim.fire_map)
        if self.sim._rendering:
            self.sim._render()
        num_updates += 1

        self.sim.elapsed_steps += 1
        
        if self.sim.config.simulation.save_data:
            self.sim._save_data()

    self.sim.active = True if self.sim.fire_status == GameStatus.RUNNING else False
    return self.sim.fire_map, self.sim.active

def run_simulation_ob(sim,total_updates):
    num_updates = 0

    while sim.fire_status == GameStatus.RUNNING and num_updates < total_updates:
        sim.fire_sprites = sim.fire_manager.sprites
        sim.fire_map, sim.fire_status = sim.fire_manager.update(sim.fire_map)
        if sim._rendering:
            sim._render()
        num_updates += 1

        sim.elapsed_steps += 1
        
        if sim.config.simulation.save_data:
            sim._save_data()

    sim.active = True if sim.fire_status == GameStatus.RUNNING else False
    return sim.fire_map, sim.active

def calc_random_start(screen_size):
    return(random.randint(0,screen_size-1),random.randint(0,screen_size-1))

def distance_to_fire(mp,x,y):
    i = -1
    mn = 2*int(mp.shape[1])
    closest_square = []
    flag = True
    for row in mp:
        i+=1
        j = -1
        for item in row:
            j+=1
            if item == 1 and np.sqrt((x-i)**2+(y-j)**2) < mn:
                flag = False
                mn = np.sqrt((x-i)**2+(y-j)**2)
                closest_square = [i,j]
    if flag:
        return 0,[0,0]
    return mn, closest_square

def distance_to_prob(mp,x,y, target):
    i = -1
    mn = 2*int(mp.shape[1])
    closest_square = []
    flag = True
    for row in mp:
        i+=1
        j = -1
        for item in row:
            j+=1
            if item == target and np.sqrt((x-i)**2+(y-j)**2) < mn:
                flag = False
                mn = np.sqrt((x-i)**2+(y-j)**2)
                closest_square = [i,j]
    if flag:
        return 0,[0,0]
    return mn, closest_square

def simple_expansion(fire_map):
    rows, cols = fire_map.shape
    new_map = copy.deepcopy(fire_map)
    for row in range(rows):
        for col in range(cols):
            if fire_map[row, col] == 1:
                for i in range(max(0, row - 1), min(row + 2, rows)):
                    for j in range(max(0, col - 1), min(col + 2, cols)):
                        if fire_map[i, j] == 0:
                            new_map[i, j] = 1
    return new_map


def generate_probabilities(self, steps):
    past_map = self.fire_map
    prob_map = np.zeros_like(past_map)
    prob_map = prob_map.astype(np.float32)
    for step in range(steps):
        new_map = simple_expansion(past_map)
        prob_map[(past_map == 0) & (new_map == 1)] = 1/(step+1)
        past_map = new_map
    return prob_map
def generate_probs_from_bench(step):
    past_map = np.load("benchmarks//"+str(step)+".npy")
    prob_map = np.zeros_like(past_map)
    prob_map = prob_map.astype(np.float32)
    for s in range(5):
        new_map = np.load("benchmarks//"+str(s+1+step)+".npy")
        prob_map[(past_map == 0) & (new_map == 1)] = 1/(s+1)
        past_map = new_map
    return prob_map
def square_state(mp, x, y):
    return mp[y][x]

class SaveModelCallback(BaseCallback):
    """
    A custom callback that saves the model.
    """
    def __init__(self, save_path, check_freq, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(os.path.join(self.save_path, f'PBP_{self.num_timesteps}'))
            self.model.save(os.path.join(self.save_path, f'PBP_{self.num_timesteps}'))
        return True
def calc_preset_start(self):
    if self.preset_fires_index > len(self.preset_fires_starts)-1:
        return self.preset_fires_starts[0], 1
    return self.preset_fires_starts[self.preset_fires_index], self.preset_fires_index + 1
def generate_benchmarks(cfg,simulation_steps_per_timestep,total_steps_per_episode):
    sim = simfire.sim.simulation.FireSimulation(cfg)
    sim.reset()
    fire_map, fire_status = run_simulation_ob(sim, 2)
    step = 0
    while True:
        if step%simulation_steps_per_timestep == 0:
            fire_map, fire_status = run_simulation_ob(sim, 1)
        np.save("benchmarks//"+str(step)+".npy",fire_map)
        if step >= total_steps_per_episode + 10:
            break
        step+=1
        

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        self.config = simfire.utils.config.Config("configs/operational_config.yml")
        self.preset_fires_starts = [(5,5),(22,5),(22,22),(5,22),(13,13),(5,13),(13,5), (22,13), (13, 22)]
        self.preset_fires_index = 0
        self.config.fire.fire_initial_position, self.preset_fires_index = calc_preset_start(self) #(13,5)
        
        self.sim = simfire.sim.simulation.FireSimulation(self.config)
        self.screen_size = self.config.area.screen_size[0]
        self.prob_map = np.zeros_like(self.sim.fire_map)
        self.fire_map, self.fire_status = run_one_simulation_step(self, 2)
        self.agent_x = 15
        self.agent_y = 15
        self.agent_start = [self.agent_x,self.agent_y]
        self.episode_steps = 0
        self.updates_per_step = 1
        self.total_steps_per_episode = 600
        self.episodes_per_fire_restart = 2500
        self.chkpt_thresh = 100
        self.simulation_steps_per_timestep = 8
        self.episode_num = 0
        self.autoplace = True
        generate_benchmarks(self.config,self.simulation_steps_per_timestep,self.total_steps_per_episode)

        self.prev_map = copy.deepcopy(self.fire_map)
        self.prev_map2 = copy.deepcopy(self.fire_map)
        self.prev_map3 = copy.deepcopy(self.fire_map)
        #---
        self.prev_prob = copy.deepcopy(self.prob_map)
        self.prev_prob2 = copy.deepcopy(self.prob_map)
        self.prev_prob3 = copy.deepcopy(self.prob_map)

        self.analytics_dir = "train_analytics//"+datetime.now().strftime("%m.%d.%Y_%H:%M:%S")
        if os.path.isdir(self.analytics_dir) == False:
            os.mkdir(self.analytics_dir)
            os.mkdir(self.analytics_dir+"//fires")
        with open(self.analytics_dir+"//customLog.txt","w") as f:
            f.write("")
        with open(self.analytics_dir+"//rewardLog.txt","w") as f:
            f.write("")

        with open(self.analytics_dir+"//customLog.txt","w") as f:
            f.write("\nENVIRONMENT GENERATED")

        if self.autoplace:
            n_actions = 4
            self.action_space = spaces.Discrete(n_actions)
            self.action_names = ["up","down","left","right"]
        else:
            n_actions = 5
            self.action_space = spaces.Discrete(n_actions)
            self.action_names = ["up","down","left","right","fireline"]
        n_channel = 2
        self.observation_space = spaces.Box(low=0, high=5,shape=(n_channel, self.screen_size, self.screen_size), dtype=np.float32)


    def step(self, action):
        self.episode_steps += 1
        action_str = self.action_names[action]
        action_multiplier = 1

        with open(self.analytics_dir+"//customLog.txt","a") as f:
            f.write("\n ACTION PREFORMED, "+action_str+","+str(self.agent_x)+","+str(self.agent_y))
        if action_str == "up":
            self.agent_y -= 1*action_multiplier
        elif action_str == "down":
            self.agent_y += 1*action_multiplier
        elif action_str == "left":
            self.agent_x -= 1*action_multiplier
        elif action_str == "right":
            self.agent_x +=1*action_multiplier
        
        if self.agent_x > self.screen_size-1:
            self.agent_x = self.screen_size-1
        if self.agent_y > self.screen_size-1:
            self.agent_y = self.screen_size-1
        if self.agent_x < 0:
            self.agent_x = 0
        if self.agent_y < 0:
            self.agent_y = 0
        
        if action_str == "fireline" and square_state(self.fire_map, self.agent_x,self.agent_y) != 1:
            self.sim.update_mitigation([(self.agent_x,self.agent_y,BurnStatus.FIRELINE)])
        
        if self.autoplace and square_state(self.fire_map, self.agent_x,self.agent_y) != 1:
            self.sim.update_mitigation([(self.agent_x,self.agent_y,BurnStatus.FIRELINE)])

        if self.episode_steps%self.simulation_steps_per_timestep == 0:
            self.fire_map, self.fire_status = run_one_simulation_step(self, self.updates_per_step)
        self.fire_map = self.sim.fire_map
        self.prob_map = generate_probabilities(self,5)


        observation_map = np.stack((self.fire_map, self.prob_map), axis=0)
        self.observation = observation_map[newaxis,:,:]
        terminated = False
        truncated = False
        if self.episode_steps > self.total_steps_per_episode:
            terminated = True
        if get_burning(self.fire_map) == 0 or not self.fire_status:
            terminated = True
            truncated = False
        reward = get_reward_bench(self.fire_map, self.prob_map,self.episode_steps)
        if square_state(self.fire_map, self.agent_x,self.agent_y) == 1:
            reward -= 100
        elif square_state(self.fire_map, self.agent_x,self.agent_y) == 2:
            reward -= 5

        with open(self.analytics_dir+"//customLog.txt","a") as f:
            f.write("\n REWARD CALCULATED, "+str(reward)+","+str(get_burned(self.fire_map))+","+str(get_burning(self.fire_map))+","+str(get_unburned(self.fire_map)))
        
        with open(self.analytics_dir+"//rewardLog.txt","a") as f:
            f.write("\n REWARD, "+str(reward)+","+str(get_burned(self.fire_map))+","+str(get_burning(self.fire_map))+","+str(get_unburned(self.fire_map))+","+str(distance_to_fire(self.fire_map,self.agent_x,self.agent_y)[0]))

        if self.chkpt_flag:
            np.save(self.chkpt_dir+"//"+str(self.episode_steps)+".npy",self.fire_map)
        if self.episode_steps%self.simulation_steps_per_timestep == 0:
            self.prev_map3 = copy.deepcopy(self.prev_map2)
            self.prev_map2 = copy.deepcopy(self.prev_map)
            self.prev_map = copy.deepcopy(self.fire_map)
        self.prev_prob3 = copy.deepcopy(self.prev_prob2)
        self.prev_prob2 = copy.deepcopy(self.prev_prob)
        self.prev_prob = copy.deepcopy(self.prob_map)
        info = {}
        return self.observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.chkpt_flag = False
        if self.episode_num % self.chkpt_thresh == 0:
            self.chkpt_flag = True
            self.chkpt_dir = self.analytics_dir+"//fires//"+str(self.episode_num)
            os.mkdir(self.chkpt_dir)
        if self.episode_num%self.episodes_per_fire_restart == 0:
            self.config.fire.fire_initial_position, self.preset_fires_index = calc_preset_start(self) #(13,22)
            generate_benchmarks(self.config,self.simulation_steps_per_timestep,self.total_steps_per_episode)
        self.sim = simfire.sim.simulation.FireSimulation(self.config)
        self.sim.reset()
        self.fire_map, self.fire_status = run_one_simulation_step(self, 2)
        self.prob_map = np.zeros_like(self.fire_map)
        observation_map = np.stack((self.fire_map, self.prob_map), axis=0)
        self.observation_return = observation_map[newaxis,:,:]
        self.episode_steps = 0
        self.agent_x = self.agent_start[0]
        self.agent_y = self.agent_start[1]
    
        self.prev_map = self.fire_map
        self.prev_map2 = self.fire_map
        self.prev_map3 = self.fire_map
        #---
        self.prev_prob = copy.deepcopy(self.prob_map)
        self.prev_prob2 = copy.deepcopy(self.prob_map)
        self.prev_prob3 = copy.deepcopy(self.prob_map)
        self.episode_num +=1

        with open(self.analytics_dir+"//customLog.txt","a") as f:
            f.write("\n NEW TRAINING ITERATION CREATION")
        with open(self.analytics_dir+"//rewardLog.txt","a") as f:
            f.write("\n NEW TRAINING ITERATION CREATION")

        info = {}
        return self.observation_return, info
    

    def render(self):
        ...

    def close(self):
        ...


from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
env = CustomEnv()
if False:
    check_env(env)
    quit()
model = DQN("MlpPolicy", env, verbose=1)
#model = PPO('MlpPolicy', env, verbose=1)

#model_path = 'previous_models//...'
#model = PPO.load(model_path, env=env)

save_path = 'saved_models//'+datetime.now().strftime("%m.%d.%Y_%H:%M:%S")
os.mkdir(save_path)
save_path += "//"
callback = SaveModelCallback(save_path=save_path, check_freq=10000)
model.learn(total_timesteps=int(8e6), progress_bar=True, callback=callback)
del model