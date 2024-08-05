import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import buffer
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.distributions import Normal
from torch.optim.lr_scheduler import LambdaLR



class CNN(nn.Module):
    def __init__(self, state_dim):
        super(CNN, self).__init__()
        
        
        self.conv1 = nn.Conv2d(state_dim[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        
        self.feature_size = self._get_feature_size(state_dim)
        self.state_norm = RunningMeanStd(self.feature_size)
        
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.norm(x, self.state_norm)
        
        return x
    
    def norm(self, x, x_norm):
        x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def _get_feature_size(self, state_dim):
        return self.conv4(self.conv3(self.conv2(self.conv1(torch.zeros(1, *state_dim))))).view(1, -1).size(1)
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                

class Actor(nn.Module):
    def __init__(self, feature_size, action_dim,ppg=False,threshold=0.5):
        super(Actor, self).__init__()
        
        self.action_dim = action_dim
        self.threshold = threshold
        self.ppg = ppg
        self.fc1 = nn.Linear(feature_size, 512)
        self.mean_linear = nn.Linear(512, action_dim)
            
        self._initialize_weights()
        
    def forward(self, x):
        
        if self.ppg:
            x = x.detach()
        x_actor = F.relu(self.fc1(x))
        mean = self.mean_linear(x_actor)
        
        
        return mean
    
    def get_action(self, state, deterministic=False):
        logits = self.forward(state) 
        probabilities = torch.sigmoid(logits)
        if deterministic:
            action = (probabilities >= self.threshold).int()
        else:
            action = torch.bernoulli(probabilities).int()
        action_log_probs = torch.log(probabilities * action + (1 - probabilities) * (1 - action))
        action_log_probs = action_log_probs.sum(dim=-1,keepdim=True)
        
        return action, action_log_probs
    
    def getLogprob(self, state, old_action):
        logits = self.forward(state) 
        probabilities = torch.sigmoid(logits)

        old_logprob = torch.log(probabilities * old_action + (1 - probabilities) * (1 - old_action) + 1e-6)
        old_logprob = old_logprob.sum(dim=-1,keepdim=True)

        distentropy = - (probabilities * torch.log(probabilities + 1e-6) + (1 - probabilities) * torch.log(1 - probabilities + 1e-6))
        distentropy = distentropy.mean() 

        return old_logprob, distentropy
    
    def _initialize_weights(self):
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'mean_linear' :
                    nn.init.orthogonal_(module.weight, 0.01)
                else:
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
                
                

class Critic(nn.Module):
    def __init__(self, feature_size):
        super(Critic, self).__init__()
        
        self.fc1_1 = nn.Linear(feature_size, 512)
        self.critic_linear = nn.Linear(512, 1)
        
        self._initialize_weights()
        
    def forward(self, x):
        
        v = F.tanh(self.fc1_1(x))
        v = self.critic_linear(v)
        
        return v
    
    def getValue(self,state):
        value= self.forward(state)
        
        return value
    
    
    def _initialize_weights(self):
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'critic_linear' :
                    nn.init.orthogonal_(module.weight, 1.0)
                else:
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('tanh'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
                
                
class SQ(nn.Module):
    def __init__(self, feature_size, action_dim):
        super(SQ, self).__init__()
        
        self.l1 = nn.Linear(feature_size + action_dim, 512)
        self.q1 = nn.Linear(512, 1)
        
        self.l2 = nn.Linear(feature_size + action_dim, 512)
        self.q2 = nn.Linear(512, 1)
        
        self._initialize_weights()
        
    def forward(self, state, action):
        
        sa = torch.cat([state, action], 1)
        q1 = F.tanh(self.l1(sa))
        q1 = self.q1(q1)
        
        q2 = F.tanh(self.l2(sa))
        q2 = self.q2(q2)
        
        return q1,q2
    
    def getQ(self,state,action):
        q1,q2 = self.forward(state,action)
        
        return q1,q2
    
    def _initialize_weights(self):
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'critic_linear' :
                    nn.init.orthogonal_(module.weight, 1.0)
                else:
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('tanh'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)



    
    

class agent(object):
    def __init__(self,state_dim, action_dim,hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.cnn_net = CNN(state_dim).to(self.device)
        self.actor = Actor(self.cnn_net.feature_size,self.action_dim,hp.ppg).to(self.device)
        self.critic = Critic(self.cnn_net.feature_size).to(self.device)
        self.Q_target = copy.deepcopy(self.critic)
        
        self.SQ = SQ(self.cnn_net.feature_size,action_dim).to(self.device)
        
        # self.actor_optimizer = torch.optim.Adam(list(self.cnn_net.parameters()) + list(self.actor.parameters()), lr=hp.actor_lr,eps=hp.eps)
        # self.critic_optimizer = torch.optim.Adam(list(self.cnn_net.parameters()) + list(self.critic.parameters()), lr=hp.critic_lr,eps=hp.eps)
        # self.sq_optimizer = torch.optim.Adam(list(self.cnn_net.parameters()) + list(self.SQ.parameters()), lr=hp.critic_lr,eps=hp.eps)

        
        param_groups = [
            {'params': self.cnn_net.parameters(), 'lr': hp.actor_lr},    # CNN 参数组
            {'params': self.actor.parameters(), 'lr': hp.actor_lr},   # Actor 参数组
            {'params': self.critic.parameters(), 'lr': hp.critic_lr}, # Critic 参数组
            {'params': self.SQ.parameters(), 'lr': hp.critic_lr}          # SQ 参数组
        ]

        self.optimizer = torch.optim.Adam(param_groups, eps=hp.eps)
        
        lambda_lr = lambda step: 1.0 - step / hp.max_steps if step < int(0.9 * hp.max_steps) else 0.1
        self.scheduler = [LambdaLR(self.optimizer, lr_lambda=lambda_lr)]
        
        #PPO
        self.clip = hp.clip
        self.grad = hp.grad
        self.num_trains = hp.num_epch_train
        self.entropy = hp.entropy
        self.c = hp.c
        self.discount = hp.discount
        self.tau = hp.tau
        #checkpoint
        self.Maxscore = (0.0,0.0)
        self.learn_step = 0
        
        
        
    @torch.no_grad()
    def select_action(self,state,deterministic=False):
        
        if state.ndim == 3:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        else:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
            
        state = self.cnn_net(state)
        action,logprob = self.actor.get_action(state,deterministic)
        value = self.critic.getValue(state)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        logprob = logprob.view(-1,1).cpu().data.numpy()
        value = value.view(-1,1).cpu().data.numpy()
        
        return action,logprob,value
    
    @torch.no_grad()
    def get_value(self,state):
        
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        state = self.cnn_net(state)
        value = self.critic.getValue(state)
        value = value.view(-1).cpu().data.numpy()
        
        return value
    
    
    
    def evaluate_actions(self, state,actions):
        
        logprob,dist_entropy = self.actor.getLogprob(state,actions)
        
        
        return logprob, dist_entropy
    
    
    def train(self,sample,process,writer):
        
        self.learn_step += 1
        states,action,old_action_log_probs,returns,next_state,mask,reward = map(lambda x: x.to(self.device), sample)
        
        
        state = self.cnn_net(states.clone())
        with torch.no_grad():
            next_state = self.cnn_net(next_state.clone())
        
        q1,q2 = self.SQ.getQ(state.clone(),action)
        
        with torch.no_grad():
            target_v = self.Q_target.getValue(next_state)
            target_q = reward + self.discount * mask * target_v
        
        q1_loss = F.mse_loss(q1,target_q)
        q2_loss = F.mse_loss(q2,target_q)
        
        q_loss = q1_loss + q2_loss
        
        
        values = self.critic.getValue(state.clone())
        value_loss = F.mse_loss(returns, values)
        
        
        action_log_probs, dist_entropy = self.evaluate_actions(state.clone(),action)
        
        
        with torch.no_grad():
            q = torch.min(q1,q2)
            v = self.critic.getValue(state.clone())
            advs = (q - v)
            advs_mean = torch.mean(advs.clone())
            advs_std = torch.std(advs.clone())
            advs = (advs - advs_mean) / (advs_std + 1e-5)
        
        ratio =  torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advs
        actor_loss = -torch.max(torch.min(surr1, surr2),self.c*advs).sum(dim=-1).mean()
        
        
        actor_loss = actor_loss - self.entropy * dist_entropy
        
        loss = q_loss + value_loss + actor_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.cnn_net.parameters()) + list(self.actor.parameters()) + 
                                       list(self.critic.parameters()) + list(self.SQ.parameters()), self.grad)
        self.optimizer.step()
        
        
        
        for scheduler in self.scheduler:
            scheduler.step()
        
        
        with torch.no_grad():
            for target_param,param in zip(self.Q_target.parameters(),self.critic.parameters()):
                target_param.data.copy_(
                target_param.data *(1 - self.tau)  + param.data * self.tau
            )
        
        writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
        writer.add_scalar('value_loss', value_loss.item(), global_step=self.learn_step)
        writer.add_scalar('q_loss', q_loss.item(), global_step=self.learn_step)
            
        if self.learn_step%self.num_trains==0:
            process.process_input(self.learn_step, 'learn_step', 'train/')
            process.process_input(actor_loss.item(), 'actor_loss', 'train/')
            process.process_input(value_loss.item(), 'value_loss', 'train/')
            process.process_input(dist_entropy.item(), 'dist_entropy', 'train/')
            process.process_input(action_log_probs.detach().cpu().numpy(), 'action_log_probs', 'train/')
            process.process_input(old_action_log_probs.detach().cpu().numpy(), 'old_action_log_probs', 'train/')
            process.process_input(ratio.detach().cpu().numpy(), 'ratio', 'train/')
            process.process_input(surr1.detach().cpu().numpy(), 'surr1', 'train/')
            process.process_input(values.detach().cpu().numpy(), 'values', 'train/')
            process.process_input(returns.detach().cpu().numpy(), 'returns', 'train/')
            
    
    
    def save(self, filename):
        torch.save({
            'cnn_state_dict': self.cnn_net.state_dict(),
            'running_mean_std_state': {
                'n': self.cnn_net.state_norm.n,
                'mean': self.cnn_net.state_norm.mean,
                'S': self.cnn_net.state_norm.S,
                'std': self.cnn_net.state_norm.std
            }
        }, filename + "_cnn")
        
        torch.save({
            'actor_state_dict': self.actor.state_dict()
        }, filename + "_actor")
        
        # 如果需要保存优化器状态，可以取消注释以下代码
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        # torch.save({
        #     'critic_state_dict': self.critic.state_dict()
        # }, filename + "_critic")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        # torch.save({
        #     'sq_state_dict': self.SQ.state_dict()
        # }, filename + "_sq")
        # torch.save(self.sq_optimizer.state_dict(), filename + "_sq_optimizer")
        
        
        
    def load(self, filename):
        cnn_checkpoint = torch.load(filename + "_cnn")
        self.cnn_net.load_state_dict(cnn_checkpoint['cnn_state_dict'])
    
        # 恢复 RunningMeanStd 的状态
        self.cnn_net.state_norm.n = cnn_checkpoint['running_mean_std_state']['n']
        self.cnn_net.state_norm.mean = cnn_checkpoint['running_mean_std_state']['mean']
        self.cnn_net.state_norm.S = cnn_checkpoint['running_mean_std_state']['S']
        self.cnn_net.state_norm.std = cnn_checkpoint['running_mean_std_state']['std']
        
        # load actor
        checkpoint = torch.load(filename + "_actor")
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        
        # 如果需要加载优化器状态，可以取消注释以下代码
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        # checkpoint = torch.load(filename + "_critic")
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        
        # checkpoint = torch.load(filename + "_sq")
        # self.SQ.load_state_dict(checkpoint['sq_state_dict'])
        # self.SQ.load_state_dict(torch.load(filename + "_sq_optimizer"))
        
        
    def IsCheckpoint(self,Score):
        """
        Args:
            Score (_type_): (fin,Score),fin代表完成了几次关卡，Score表示总体表现情况

        Returns:
            _type_: 是否是最好的模型
        """
        if self.Maxscore[0]<Score[0]:
            self.Maxscore = Score
            return True
        elif self.Maxscore[0]==Score[0]:
            if self.Maxscore[1]<Score[1]:
                self.Maxscore = Score
                return True
            else:
                return False
        else:
            return False
                
                
                
                
            
            
            
            
            
            
            
            
        
        
        
        