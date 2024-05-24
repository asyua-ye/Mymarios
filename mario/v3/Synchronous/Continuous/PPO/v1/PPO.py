import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import buffer
import numpy as np
from torch.distributions import Normal
import math


        
    
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
        else:
            log_prob = None
            
        action = self.max_action*action
        
        
        return action,log_prob,value,z
    
    def getValue(self,state):
        mean, log_std, value= self.forward(state)
        
        return value
    
    def getLogprob(self,state,action,old_z):
        
        mean, log_std, value= self.forward(state)
        
        std = log_std.exp()
        normal = Normal(mean, std)
        action = action / self.max_action
        
        old_logprob = normal.log_prob(old_z) - torch.log(1 - action.pow(2) + 1e-6)
        
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
    def __init__(self,state_dim, action_dim,max_action,hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actorCritic = ActorCritic(self.state_dim,self.action_dim,max_action,hp.log_std_min,hp.log_std_max).to(self.device)
        self.actorCritic_o = torch.optim.Adam(self.actorCritic.parameters(),lr=hp.actor_lr)
        self.replaybuffer = buffer.ReplayBuffer(hp.buffer_size,hp.num_processes,hp.num_steps)
        
        
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
        
        if state.ndim == 3:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        else:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        action,logprob,value,z = self.actorCritic.getAction(state,deterministic)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        logprob = logprob.view(-1,self.action_dim).cpu().data.numpy()
        value = value.view(-1,1).cpu().data.numpy()
        z = z.view(-1,self.action_dim).cpu().data.numpy()
        return action,logprob,value,z
    
    @torch.no_grad()
    def get_value(self,state):
        
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        value = self.actorCritic.getValue(state)
        value = value.view(-1).cpu().data.numpy()
        
        return value
    
    
    
    def evaluate_actions(self, state,actions,old_z):
        
        logprob,value,dist_entropy = self.actorCritic.getLogprob(state,actions,old_z)
        
        return value, logprob, dist_entropy
    
    
    def train(self):
        
        value_loss_epoch = 0
        actor_loss_epoch = 0
        
        for i in range(self.ppo):
            data_generator = self.replaybuffer.PPOsample(self.num_mini_batch)
            
            for sample in data_generator:
                state,action,old_action_log_probs,returns,advs,old_z = sample
                
                values, action_log_probs, dist_entropy = self.evaluate_actions(state,action,old_z)
                
                ratio =  torch.exp(action_log_probs - old_action_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advs
                actor_loss = -torch.min(surr1, surr2).sum(dim=-1).mean()
                
                
                value_loss = F.mse_loss(returns.mean(-1,keepdim=True), values)
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
                
                
                
                
            
            
            
            
            
            
            
            
        
        
        
        
