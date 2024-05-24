import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import buffer
import numpy as np
from torch.distributions import Normal
import math
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    # Generic
    buffer_size: int = int(2560)
    discount: float = 0.99
    gae: float = 0.95
    grad: float = 0.5
    
    
    #Actor
    actor_lr: float = 3e-4
    entropy: float = 0.01
    log_std_max: int = 2
    log_std_min: int = -20
    
    #Critic
    critic_lr: float = 3e-4
    
    
    #PPO
    clip: float = 0.2
    ppo_update: int = 50
    mini_batch: int = 20
    value: float = 0.5
    actor: float = 1.0
    
    #epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_final: float = 0.01
    epsilon_decay: int = 60000
        
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,max_action,log_std_min=-20,log_std_max=2):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action
        
        self.conv1 = nn.Conv2d(self.state_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.critic_linear = nn.Linear(512, 1)
        
        self.fc2 = nn.Linear(self.feature_size(), 512)
        self.mean_linear = nn.Linear(512, action_dim)
        self.log_std_linear = nn.Linear(512, action_dim)  
        
        self._initialize_weights()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        v = F.relu(self.fc1(x))
        value = self.critic_linear(v)
        
        a = F.relu(self.fc2(x))
        mean    = self.mean_linear(a)
        log_std = self.log_std_linear(a).clamp(self.log_std_min, self.log_std_max)
        
        return mean, log_std, value
    
    def getAction(self,state,deterministic=False,with_logprob=True,rsample=True):
        
        mean, log_std, value= self.forward(state)
        
        
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        if deterministic:
            z = mean
        else:
            if rsample:
                z = normal.rsample()
            else:
                z = normal.sample()
                
                
                
        action = torch.sigmoid(z)
        
        if with_logprob:
            log_prob = normal.log_prob(z)
            # log_prob -= (2 * (np.log(2) - z - F.softplus(-2 * z)))
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            log_prob = None
            
        action = self.max_action*action
        
        
        return action,log_prob,value
    
    def getValue(self,state):
        mean, log_std, value= self.forward(state)
        
        return value
    
    def getLogprob(self,state,action,deterministic=False,rsample=True):
        
        mean, log_std, value= self.forward(state)
        
        std = log_std.exp()
        normal = Normal(mean, std)
        
        if deterministic:
            z = mean
        else:
            if rsample:
                z = normal.rsample()
            else:
                z = normal.sample()
        
        
        
        action = action / self.max_action
        # old_z = torch.atanh(torch.clamp(action, min=-0.999999, max=0.999999))
        
        old_logprob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        old_logprob = old_logprob.sum(-1, keepdim=True)
        
        distentroy= 0.5 * torch.log(2 * torch.pi * torch.e * std**2).mean()
        
        return old_logprob,value,distentroy
        

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.state_dim)))).view(1, -1).size(1)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    

class agent(object):
    def __init__(self,state_dim, action_dim,max_action,hp=Hyperparameters()) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actorCritic = ActorCritic(self.state_dim,self.action_dim,max_action,hp.log_std_min,hp.log_std_max).to(self.device)
        self.actorCritic_o = torch.optim.Adam(self.actorCritic.parameters(),lr=hp.actor_lr)
        
        self.replaybuffer = buffer.ReplayBuffer(state_dim,action_dim,hp.discount,hp.gae,hp.buffer_size)
        
        
        self.learn_step_counter = 0
        self.epsilon_by_frame = lambda frame_idx: hp.epsilon_final + (hp.epsilon_start - hp.epsilon_final) * math.exp(-1. * frame_idx / hp.epsilon_decay)
        self.epsilon = self.epsilon_by_frame(self.learn_step_counter)
        
        #PPO
        self.ppo = hp.ppo_update
        self.clip = hp.clip
        self.grad = hp.grad
        self.num_mini_batch = hp.mini_batch
        self.entropy = hp.entropy
        self.value = hp.value
        self.actor = hp.actor
        
        #checkpoint
        self.Maxscore = 0
        
        
        
    @torch.no_grad()
    def select_action(self,state,deterministic=False):
        
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        action,logprob,value = self.actorCritic.getAction(state,deterministic)
        action = action.view(-1,self.action_dim).cpu().data.numpy()[0]
        logprob = logprob.view(-1,1).cpu().data.numpy()
        value = value.view(-1,1).cpu().data.numpy()
        return action,logprob,value
    
    @torch.no_grad()
    def get_value(self,state):
        
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        value = self.actorCritic.getValue(state)
        value = value.view(-1).cpu().data.numpy()
        
        return value
    
    
    
    def evaluate_actions(self, state,actions):
        
        logprob,value,dist_entropy = self.actorCritic.getLogprob(state,actions)
        
        return value, logprob, dist_entropy
    
    
    def train(self):
        
        value_loss_epoch = 0
        actor_loss_epoch = 0
        
        for i in range(self.ppo):
            data_generator = self.replaybuffer.PPOsample(self.num_mini_batch)
            
            for sample in data_generator:
                state,action,returns,old_action_log_probs,advs = sample
                
                self.learn_step_counter +=1
                
                values, action_log_probs, dist_entropy = self.evaluate_actions(state,action)
                
                ratio =  torch.exp(action_log_probs - old_action_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advs
                actor_loss = -torch.min(surr1, surr2).mean()
                
                
                value_loss = F.mse_loss(returns, values)
                actor_loss = self.actor * actor_loss - self.entropy * dist_entropy
                loss = actor_loss + self.value * value_loss
                
                
                self.actorCritic_o.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actorCritic.parameters(), self.grad)
                self.actorCritic_o.step()
                
                value_loss_epoch += value_loss.item()
                actor_loss_epoch += actor_loss.item()
                
        
        value_loss_epoch /= (self.ppo * self.num_mini_batch)
        actor_loss_epoch /= (self.ppo * self.num_mini_batch)
        
        return  actor_loss_epoch,value_loss_epoch 
    
    
    def save(self,filename):
        torch.save(self.actorCritic.state_dict(),filename+"_actorCritic")
        torch.save(self.actorCritic_o.state_dict(),filename+"_actorCritic_optim")
        
        
        
    def load(self,filename):
        self.actorCritic.load_state_dict(torch.load(filename+"_actorCritic"))
        self.actorCritic_o.load_state_dict(torch.load(filename+"_actorCritic_optim"))
        
    def IsCheckpoint(self,Score):
        if self.Maxscore<Score:
            self.Maxscore = Score
            return True
        else:
            return False
                
                
                
                
            
            
            
            
            
            
            
            
        
        
        
        
