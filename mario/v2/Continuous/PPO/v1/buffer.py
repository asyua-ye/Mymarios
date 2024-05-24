import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



class ReplayBuffer(object):
    def __init__(self, state_dim,action_dim,discount=0.99,gae=0.95,max_size=int(1e3)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size,) + state_dim)
        self.action = np.zeros((max_size, action_dim))
        self.returns = np.zeros((max_size, 1))
        self.logits = np.zeros((max_size, 1))
        self.adv = np.zeros((max_size, 1))
        
        self.rewards = np.zeros((max_size, 1))
        self.values = np.zeros((max_size, 1))
        self.masks = np.zeros((max_size, 1))
        
        self.discount = discount
        self.gae_tau = gae
        self.flag = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def flagtoFalse(self):
        self.flag = False


    def add(self, state, action, value, reward, done, logit):
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.values[self.ptr] = value
        self.masks[self.ptr] = 1 - done
        self.rewards[self.ptr] = reward
        self.logits[self.ptr] = logit
        
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.ptr==0:
            self.flag = True
            
        return self.flag
        


    def computeReturn(self,next_value,done):
        """
        每次和环境交互完了用
        
        """
        mask = 1 - done
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            
            delta= self.rewards[step] + next_value * self.discount * mask- self.values[step]
            gae = delta+self.discount * self.gae_tau * mask * gae
            self.returns[step] = gae + self.values[step]
            
            next_value = self.values[step]
            mask = self.masks[step]
            
            
        self.adv = self.returns - self.values
        
        self.adv = (self.adv - np.mean(self.adv)) / (
            np.std(self.adv) + 1e-5)
        
        
        
    def PPOsample(self, num_mini_batch=40):
        """
        因为ppo是onpolicy的，所以它一次要把整个回放池的东西都过一遍
        但是每次取的都是随机的样本，不过是不放回的
        """
        
        mini_batch_size = self.max_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(self.max_size)), mini_batch_size, drop_last=False)
        
        for ind in sampler:
            
            yield (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.returns[ind]).to(self.device),
                torch.FloatTensor(self.logits[ind]).to(self.device),
                torch.FloatTensor(self.adv[ind]).to(self.device)   
            )
            
    def sample(self,batch_size):
        
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.returns[ind]).to(self.device)
        )