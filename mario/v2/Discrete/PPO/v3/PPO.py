import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import buffer
import math
from dataclasses import dataclass
from typing import Callable

@dataclass
class Hyperparameters:
    # Generic
    buffer_size: int = int(10240)
    discount: float = 0.9
    gae: float = 1.0
    grad: float = 0.5
    
    
    #Actor
    actor_lr: float = 3e-4
    entropy: float = 0.01
    activa: Callable = F.relu
    hidden_size: int = 512
    
    #Critic
    critic_lr: float = 3e-4
    
    
    #PPO
    clip: float = 0.2
    ppo_update: int = 30
    mini_batch: int = 40
    value: float = 0.5
    actor: float = 1.0
    
    #epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_final: float = 0.01
    epsilon_decay: int = 60000
    
    
    
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, activa=nn.ReLU, hidden_size=512):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activa = activa
        self.hidden_size = hidden_size
        
        self.conv1 = nn.Conv2d(self.state_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lstm = nn.LSTMCell(self.feature_size(), self.hidden_size)
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, self.action_dim)
        
        self._initialize_weights()


    def forward(self, x, hidden):
        x = self.activa(self.conv1(x))
        x = self.activa(self.conv2(x))
        x = self.activa(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x, cell = self.lstm(x,hidden)
        value = self.critic_linear(x)
        logits = self.actor_linear(x)

        return logits, value, (x,cell)

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.state_dim)))).view(1, -1).size(1)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.orthogonal_(module.weight_ih)
                nn.init.orthogonal_(module.weight_hh)
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)
                forget_gate_bias = 1
                nn.init.constant_(module.bias_ih[self.hidden_size:2*self.hidden_size], forget_gate_bias)
                nn.init.constant_(module.bias_hh[self.hidden_size:2*self.hidden_size], forget_gate_bias)
    
    

class agent(object):
    def __init__(self,state_dim, action_dim,test = False,hp=Hyperparameters()) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.test = test
        self.actorCritic = ActorCritic(self.state_dim,self.action_dim,hp.activa,hp.hidden_size).to(self.device)
        self.actorCritic_o = torch.optim.Adam(self.actorCritic.parameters(),lr=hp.actor_lr)
        self.replaybuffer = buffer.ReplayBuffer(state_dim,hp.discount,hp.gae,hp.buffer_size)
        
        
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
        self.actor_w = hp.actor
        
        #checkpoint
        self.Maxscore = 0
        
        
        
    @torch.no_grad()
    def select_action(self,state,hidden):
        
        
        hidden = (torch.from_numpy(hidden[0].reshape(1,-1)).to(self.device),torch.from_numpy(hidden[1].reshape(1,-1)).to(self.device))
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        logits,value, hidden = self.actorCritic(state,hidden)
        
        if self.test:
            logits = torch.where(logits > 0, logits + 1, logits.exp())
            probs = logits / logits.sum(dim=1, keepdim=True)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().view(-1, 1)
            action = action.cpu().data.numpy()[0][0]
            return action
        else:
            
            logits = torch.where(logits > 0, logits + 1, logits.exp())
            probs = logits / logits.sum(dim=1, keepdim=True)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().view(-1, 1)
            action_log_probs = dist.log_prob(action.squeeze(-1)).view(-1, 1)
            action = action.cpu().data.numpy()[0][0]
            
                

        return (
                action, 
                action_log_probs.view(-1).detach().cpu().numpy(),
                value.view(-1).detach().cpu().numpy(),
                (hidden[0].view(-1).detach().cpu().numpy(), hidden[1].view(-1).detach().cpu().numpy())
                )
    
    @torch.no_grad()
    def get_value(self,state,hidden):
        
        hidden = (torch.from_numpy(hidden[0].reshape(1,-1)).to(self.device),torch.from_numpy(hidden[1].reshape(1,-1)).to(self.device))
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        _,value,_ = self.actorCritic(state,hidden)
        
        
        return value.view(-1).detach().cpu().numpy()
    
    
    
    def evaluate_actions(self, state, actions, hidden):
        
        logits,values,_ = self.actorCritic(state,hidden)
        
        logits = torch.where(logits > 0, logits + 1, logits.exp())
        probs = logits / logits.sum(dim=1, keepdim=True)
        dist = torch.distributions.Categorical(probs=probs)
        action_log_probs = dist.log_prob(actions.squeeze(-1)).view(-1, 1)
        dist_entropy = dist.entropy().mean()
        
        
        return values, action_log_probs, dist_entropy
    
    
    def train(self):
        
        value_loss_epoch = 0
        actor_loss_epoch = 0
        
        for i in range(self.ppo):
            data_generator = self.replaybuffer.PPOsample(self.num_mini_batch)
            
            for sample in data_generator:
                state,action,returns,old_action_log_probs,advs,hidden0,hidden1 = sample
                
                self.learn_step_counter +=1
                
                values, action_log_probs, dist_entropy = self.evaluate_actions(state,action,(hidden0,hidden1))
                
                ratio =  torch.exp(action_log_probs - old_action_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advs
                actor_loss = -torch.min(surr1, surr2).mean()
                
                
                value_loss = F.mse_loss(returns, values)
                actor_loss = self.actor_w * actor_loss - self.entropy * dist_entropy
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
                
                
                
                
            
            
            
            
            
            
            
            
        
        
        
        
