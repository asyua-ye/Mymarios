import numpy as np
import gymnasium as gym
import cv2
cv2.ocl.setUseOpenCL(False)
from joypad_space import myJoypadSpace
from gymnasium.spaces import Box
from gymnasium.wrappers import LazyFrames

MYACTION = [
    ['NOOP'],
    ['right'],
    ['left'],
    ['down'],
    ['up'],
    ['A'],
    ['B']
]


def Reward(info,reward,old_info,done):
    
    p = 0
    # 计算累计score
    s = (info['score'] - old_info['score'])/10
    
    # 是否到达终点
    Flag = info['flag_get']
    if Flag:
        p = info['score']
    
    # 是否死亡
    if done and not Flag:
        p = -info['score']
    
    reward += p + s
        
    return reward        




class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip,lz4_compress = False):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.action_dim = env.action_space.n
        self.Frames = np.zeros((1, env.action_space.n))
        self.lz4_compress = lz4_compress
        obs_shape = env.observation_space.shape
        obs_dtype = env.observation_space.dtype

        # 新的观测空间应该是堆叠的帧
        new_shape = (skip,) + obs_shape  # 添加帧数维度
        # 更新观测空间
        self.observation_space = Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=obs_dtype
        )
        
        action_shape = env.action_space.n
        action_dtype = env.action_space.dtype
        new_low = np.full(action_shape, -1, dtype=action_dtype)
        new_high = np.full(action_shape, 1, dtype=action_dtype)
        
        self.action_space = gym.spaces.Box(low=new_low, high=new_high, dtype=action_dtype)
        self.old_info = None
        
        
    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(observation) == self._skip, (len(observation), self._skip)
        return LazyFrames(observation, self.lz4_compress)

    def step_one(self, action):
        
    
        """
        这个是规划中途可以改变的版本
        这里留下了一个错误，这里每次只执行了1帧
        skips = np.min(self.Frames)
        
        """
        total_reward = 0.0
        
        Frames = action * 60
        rounded_Frames = np.round(Frames)
        self.Frames += rounded_Frames
        self.Frames = np.where(self.Frames < 0,0,self.Frames)
        a = np.nonzero(self.Frames[0])[0]
        real_a = [MYACTION[i] for i in a]
        non_zero_frames = self.Frames[self.Frames > 0]
        if non_zero_frames.size > 0:
            skips = np.min(non_zero_frames)
        else:
            skips = 0
        self.Frames -= skips
        self.Frames = np.where(self.Frames < 0,0,self.Frames)
        skips = int(skips)
        if skips < self._skip:
            skips = self._skip
            random_values = list(range(skips))
        else:
            available_indices = list(range(1, skips-1))
            random_values = np.random.choice(available_indices, self._skip - 2, replace=False)
            random_values = np.sort(random_values)
        
        obs_all = []
        obs_temp = []
        total_reward = 0
        old = 0
        for i in range(skips):
            obs, reward, done, trunk, info = self.env.step(real_a)
            if self.old_info is None:
                self.old_info = info
            reward = Reward(info,reward,self.old_info,done)
            self.old_info = info
            total_reward += reward
            if i == 0 or i == skips - 1 or i in random_values:
                obs_all.append(obs)
                old = i
            obs_temp.append(obs)
            if done:
                self.old_info = None
                self.Frames = np.zeros((1, self.action_dim))
                if len(obs_all) < self._skip:
                    num = self._skip - 1 - len(obs_all)
                    if len(obs_temp) > 1 and old < len(obs_temp) - 1:
                        # 确保有可选择的范围
                        available_range = list(range(old, len(obs_temp)-1))
                        if len(available_range) < num:
                            # 如果可选范围内的元素不足，选择全部可用元素，允许重复
                            ind = np.random.choice(available_range, num, replace=True)
                        else:
                            # 正常情况，不重复选择
                            ind = np.random.choice(available_range, num, replace=False)
                        ind = np.sort(ind)  # 对索引排序
                        obs_all.extend([obs_temp[i] for i in ind])
                    else:
                        last_obs = obs_temp[-1] if obs_temp else None
                        num = self._skip - 1 - len(obs_all)
                        if num > 0:
                            obs_all.extend([last_obs] * num)
                        
                    if obs_temp:
                        obs_all.append(obs_temp[-1])
                break
        return self.observation(obs_all), total_reward, done, trunk, info
    
    def step(self,action):
        
        """
        这个是规划了不能变的版本
        
        """
        total_reward = 0.0
        
        Frames = action * 60
        
        Frames = np.where(Frames < 0, 0, Frames)
        rounded_Frames = np.round(Frames)
        self.Frames += rounded_Frames
        non_zero_frames = self.Frames[self.Frames > 0]
        if non_zero_frames.size > 0:
            skips = np.max(non_zero_frames)
        else:
            skips = 0
        skips = int(skips)
        if skips < self._skip:
            skips = self._skip
            random_values = list(range(skips))
        else:
            available_indices = list(range(1, skips-1))
            random_values = np.random.choice(available_indices, self._skip - 2, replace=False)
            random_values = np.sort(random_values)
        
        obs_all = []
        obs_temp = []
        total_reward = 0
        old = 0
        for i in range(skips):
            a = np.nonzero(self.Frames[0])[0]
            real_a = [MYACTION[i] for i in a]
            self.Frames -= 1
            self.Frames = np.where(self.Frames < 0,0,self.Frames)
            obs, reward, done, trunk, info = self.env.step(real_a)
            
            if self.old_info is None:
                self.old_info = info
            reward = Reward(info,reward,self.old_info,done)
            self.old_info = info
            
            total_reward += reward
            if i == 0 or i == skips - 1 or i in random_values:
                obs_all.append(obs)
                old = i
            obs_temp.append(obs)
            if done:
                self.old_info = None
                if len(obs_all) < self._skip:
                    num = self._skip - 1 - len(obs_all)
                    if len(obs_temp) > 1 and old < len(obs_temp) - 1:
                        # 确保有可选择的范围
                        available_range = list(range(old, len(obs_temp)-1))
                        if len(available_range) < num:
                            # 如果可选范围内的元素不足，选择全部可用元素，允许重复
                            ind = np.random.choice(available_range, num, replace=True)
                        else:
                            # 正常情况，不重复选择
                            ind = np.random.choice(available_range, num, replace=False)
                        ind = np.sort(ind)  # 对索引排序
                        obs_all.extend([obs_temp[i] for i in ind])
                    else:
                        last_obs = obs_temp[-1] if obs_temp else None
                        num = self._skip - 1 - len(obs_all)
                        if num > 0:
                            obs_all.extend([last_obs] * num)
                        
                    if obs_temp:
                        obs_all.append(obs_temp[-1])
                break
        self.Frames = np.zeros((1, self.action_dim))
        return self.observation(obs_all), total_reward, done, trunk, info
        
    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs_all = []
        obs, info = self.env.reset(**kwargs)
        
        self.old_info = None
        self.Frames = np.zeros((1, self.action_dim))

        [obs_all.append(obs) for _ in range(self._skip)]

        return self.observation(obs_all), info


    
    
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84,skip=4, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        self._skip = skip
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._skip, self._height, self._width),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
            
        assert original_space.dtype == np.uint8

    def observation(self, obs):
        obs_all = []  # 初始化存放所有处理后的观察结果的列表
        new_obs = []  # 用于存储修改后的观察结果，当 self._key 不为 None 时使用

        for ob in obs:
            if self._key is None:
                frame = ob
            else:
                frame = ob[self._key]
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if self._grayscale:
                    # 转换颜色空间为灰度
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # 调整图像大小
                frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
                if self._key is None:
                    obs_all.append(frame)
                else:
                    new_ob = ob.copy()
                    new_ob[self._key] = frame
                    new_obs.append(new_ob)
        if self._key is not None:
            return new_obs
        else:
            return obs_all
    
    
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0
    
    
def ProcessEnv(env,N=4):
    env = myJoypadSpace(env, MYACTION)
    env = SkipFrame(env, skip=N)
    env = WarpFrame(env,skip=N)
    env = ScaledFloatFrame(env)
    return env
    
    
    

