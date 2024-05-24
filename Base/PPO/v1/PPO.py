import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import buffer
import math
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    # Generic
    buffer_size: int = int(10240)
    discount: float = 0.99
    gae: float = 0.95
    grad: float = 0.5
    
    
    #Actor
    actor_lr: float = 3e-4
    entropy: float = 0.01
    
    #Critic
    critic_lr: float = 3e-4
    
    
    #PPO
    clip: float = 0.2
    ppo_update: int = 10
    mini_batch: int = 40
    value: float = 0.5
    actor: float = 1.0
    
    #epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_final: float = 0.01
    epsilon_decay: int = 60000
    
    
    
class Actor(nn.Module):
    def __init__(self, state_dim,action_dim):
        super(Actor, self).__init__()
        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0),
                    nn.init.calculate_gain('relu'))
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.conv1 = nn.Conv2d(self.state_dim[0],32,kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        
        
        self.fc1 = init_(nn.Linear(self.feature_size(), 512))
        
        
        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0), gain=0.01)
        
        self.fc2 = init_(nn.Linear(512, self.action_dim))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits
    
    def layer_init(self,module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data,gain=gain)
        bias_init(module.bias.data)
        return module
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1,*self.state_dim)))).view(1, -1).size(1)
    
    
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0),
                    nn.init.calculate_gain('relu'))
        
        self.state_dim = state_dim
        self.conv1 = nn.Conv2d(self.state_dim[0],32,kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        
        self.fc1 = init_(nn.Linear(self.feature_size(), 512))
        
        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0), gain=0.01)
        
        self.fc2 = init_(nn.Linear(512, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)

        return value
    
    def layer_init(self,module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data,gain=gain)
        bias_init(module.bias.data)
        return module
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1,*self.state_dim)))).view(1, -1).size(1)
    
    
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.conv1 = nn.Conv2d(self.state_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(), 512)


        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, self.action_dim)
        
        self._initialize_weights()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        value = self.critic_linear(x)
        logits = self.actor_linear(x)

        return logits, value

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
    def __init__(self,state_dim, action_dim,test = False,hp=Hyperparameters()) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.test = test
        # self.actor = Actor(self.state_dim,self.action_dim).to(self.device)
        # self.critic = Critic(self.state_dim).to(self.device)
        # self.actor_o = torch.optim.Adam(self.actor.parameters(),lr=hp.actor_lr)
        # self.critic_o = torch.optim.Adam(self.critic.parameters(),lr=hp.critic_lr)
        self.actorCritic = ActorCritic(self.state_dim,self.action_dim).to(self.device)
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
        self.actor = hp.actor
        
        #checkpoint
        self.Maxscore = 0
        
        
        
    @torch.no_grad()
    def select_action(self,state):
        
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        # logits = self.actor(state)
        # value = self.critic(state)
        logits,value = self.actorCritic(state)
        
        if self.test:
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = F.log_softmax(logits, dim=1)
            action = dist.sample().view(-1, 1)
            action_log_probs = log_probs.gather(1, action.long())
            action = action.cpu().data.numpy()[0][0]
            return action,0,1
        else:
            
            """
            当logits是负的时候，会出问题
            有正有负的时候
            
            上面的情况，只能用softmax
            
            全是正的时候不会出问题
            
            
            """            
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = F.log_softmax(logits, dim=1)
            action = dist.sample().view(-1, 1)
            action_log_probs = log_probs.gather(1, action.long())
            action = action.cpu().data.numpy()[0][0]
            
                

        return action,action_log_probs.view(-1).cpu().data.numpy(),value.view(-1).cpu().data.numpy()
    
    @torch.no_grad()
    def get_value(self,state):
        
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        # values = self.critic(state)
        logits,value = self.actorCritic(state)
        
        
        return value.view(-1).cpu().data.numpy()
    
    
    
    def evaluate_actions(self, state,actions):
        
        # logits = self.actor(state)
        # values = self.critic(state)
        logits,values = self.actorCritic(state)
        
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = F.log_softmax(logits, dim=1)
        
        action_log_probs = log_probs.gather(1, actions.long())
        dist_entropy = dist.entropy().mean()
        
        
        return values, action_log_probs, dist_entropy
    
    
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
                
                # self.actor_o.zero_grad()
                # actor_loss.backward()
                # self.actor_o.step()
                
                # self.critic_o.zero_grad()
                # value_loss.backward()
                # self.critic_o.step()
                
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
        # torch.save(self.actor.state_dict(),filename+"_actor")
        # torch.save(self.actor_o.state_dict(),filename+"_actor_optim")
        # torch.save(self.critic.state_dict(),filename+"_critic")
        # torch.save(self.critic_o.state_dict(),filename+"_critic_optim")
        
        
        
    def load(self,filename):
        self.actorCritic.load_state_dict(torch.load(filename+"_actorCritic"))
        self.actorCritic_o.load_state_dict(torch.load(filename+"_actorCritic_optim"))
        # self.actor.load_state_dict(torch.load(filename+"_actor"))
        # self.actor_o.load_state_dict(torch.load(filename+"_actor_optim"))
        # self.critic.load_state_dict(torch.load(filename+"_critic"))
        # self.critic_o.load_state_dict(torch.load(filename+"_critic_optim"))
        
    def IsCheckpoint(self,Score):
        if self.Maxscore<Score:
            self.Maxscore = Score
            return True
        else:
            return False
                
                
                
                
            
            
            
            
            
            
            
            
        
        
        
        
