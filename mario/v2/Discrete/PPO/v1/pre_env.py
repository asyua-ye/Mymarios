import numpy as np
import gymnasium as gym
import cv2
cv2.ocl.setUseOpenCL(False)
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT,RIGHT_ONLY


def Reward(info,reward,old_info,done):
    
    p = 0
    # 计算累计score
    s = (info['score'] - old_info['score'])/10
    s = np.clip(s, -5, 5)
    
    #时间惩罚
    t = (info['time'] - old_info['time'])*2
    if t > 0:
        t = 0
    
    
    #向右奖励
    x = (info['x_pos'] - old_info['x_pos'])
    if x >= 0:
        x = 0
    elif x < 0:
        x = x * 20
    elif x==0:
        x = -5
    
    
    # 是否到达终点
    Flag = info['flag_get']
    if Flag:
        p = info['score']
    
    # 是否死亡
    if done and not Flag:
        p = -info['score']
    
    reward += (p + s + t + x)
    
    reward /= 4.0 
        
    return reward        

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.old_info = None

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            
            if self.old_info is None:
                self.old_info = info
            reward = Reward(info,reward,self.old_info,done)
            self.old_info = info
            
            total_reward += reward
            if done:
                self.old_info = None
                break
        return obs, total_reward, done, trunk, info
    
    def reset(self, **kwargs):
        
        self.old_info = None

        return self.env.reset(**kwargs)


    
    
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        
        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
            
        
        return obs
    
    
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0
    
    
def ProcessEnv(env):
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)
    return env
    
    
    

