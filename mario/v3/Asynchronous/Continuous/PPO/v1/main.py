import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import argparse
import os
import time
import datetime
import PPO
from buffer import RolloutBuffer
from pre_env import ProcessEnv
from dataclasses import dataclass, asdict
from subproc_vec_env import SubprocVecEnv


"""
异步采样


一个经典的生产者消费者的问题，actor不停的生产，learner不停的消费
最理想的情况是，两个角色都自己干自己的，几乎不停。

actor,会描述自己的采样，每次读取新的模型参数(读取200次就结束采样)，有个提示
learner，就闷头训练就可以了

"""

@dataclass
class Hyperparameters:
    # Generic
    buffer_size: int = 1
    discount: float = 0.65
    gae: float = 0.2
    grad: float = 0.5
    num_processes: int = 8
    num_steps: int = 600
    device: torch.device = None
    
    # Actor
    actor_lr: float = 3e-4
    entropy: float = 0.01
    log_std_max: int = 2
    log_std_min: int = -20
    
    # Critic
    critic_lr: float = 3e-4
    
    # PPO
    clip: float = 0.25
    ppo_update: int = 40
    mini_batch: int = 40
    value: float = 0.5
    actor: float = 1.0
    
    # RL
    env: str = "SuperMarioBros-1-1-v0"
    seed: int = 0
    test: bool = False

    # Evaluation
    checkpoint: bool = True
    eval_eps: int = 7
    max_timesteps: int = 200
    eval: int = 1

    # File
    file_name: str = None

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def save_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for key, value in asdict(self).items():
                f.write(f"{key}: {value}\n")
    



def train_online(RL_agent, env, eval_env, rollout, hp, args):
    """
    不考虑单进程的情况了
    
    
    """
    state = env.reset()
    episode_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    final_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    rounds = np.zeros(hp.num_processes, dtype=np.float64)
    start_time = time.time()
    file_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    fo = (f"begin time:  {file_time}\n")
    writer = SummaryWriter(f'./{args.file_name}/output/{file_time}/{file_time}-PPO-{args.max_timesteps}')
    if args.checkpoint and not os.path.exists(f"./{args.file_name}/output/{file_time}/checkpoint/models"):
        os.makedirs(f"./{args.file_name}/output/{file_time}/checkpoint/models")
    train = []
    train.append(fo)
    if not os.path.exists(f"./{args.file_name}/output/{file_time}/models/"):
        output_directory = f'./{args.file_name}/output/{file_time}/models/'
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(output_directory, 'hyperparameters.txt'))
    
    for t in range(int(args.max_timesteps+1)):
        
        for step in range(hp.num_steps):
            rounds += 1
            action,logit,value,z = RL_agent.select_action(np.array(state))
            next_state, reward, ep_finished, _ = env.step(action)
            if np.any(reward!=0):
                episode_rewards += np.max(reward[reward != 0], axis=-1)
            mask =  1. - ep_finished.astype(np.float32)
            final_rewards *= mask
            final_rewards += (1. - mask) * episode_rewards
            episode_rewards *= mask
            
            reward = torch.from_numpy(reward.astype(np.float32)).to(hp.device)
            mask = torch.from_numpy(mask).to(hp.device).view(-1, 1)
            state = torch.from_numpy(state.astype(np.float64)).to(hp.device)
            action = torch.from_numpy(action.astype(np.float64)).to(hp.device)
            logit = torch.from_numpy(logit).to(hp.device)
            value = torch.from_numpy(value).to(hp.device)
            z = torch.from_numpy(z).to(hp.device)
                       
            rollout.insert(state, action,logit, value, reward, mask,z)
            state = next_state
            if torch.any(mask == 0).item() and np.any(final_rewards != 0):
                non_zero_rewards = final_rewards[final_rewards != 0]
                writer.add_scalar('reward', np.mean(non_zero_rewards), global_step=t)
                fo = (
                    f"T: {t} Total T: {np.sum(rounds)}  mean: {np.mean(non_zero_rewards):.3f} "
                    f"mid: {np.median(non_zero_rewards):.3f} max: {np.max(non_zero_rewards):.3f} "
                    f"min: {np.min(non_zero_rewards):.3f}"
                        )
                print(fo)
                train.append(fo)
                
        fo = (f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")
        print(fo)
        train.append(fo)
        next_value = RL_agent.get_value(np.array(next_state))
        next_value = torch.from_numpy(next_value).view(-1,1).to(hp.device)
        states,actions,action_log_probs,advs,zs,returns = rollout.computeReturn(next_value,mask)
        data=(np.copy(states), np.copy(actions),np.copy(action_log_probs), np.copy(advs),np.copy(zs),np.copy(returns))
        RL_agent.replaybuffer.push(data)
        fo = (f"T: {t}   sample end begintrain！！")
        print(fo)
        train.append(fo)
        actor_loss,value_loss = RL_agent.train()
        writer.add_scalar('actor_loss', actor_loss, global_step=t)
        writer.add_scalar('value_loss', value_loss, global_step=t)
        # evals = []
        # if args.checkpoint:
        #     maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time,file_time, args)
        np.savetxt(f"./{args.file_name}/output/{file_time}/train.txt", train, fmt='%s')
        RL_agent.save(f'./{args.file_name}/output/{file_time}/models/')
        
    env.close()
    file_time1 = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    fo = (f"\nend time: {file_time1}")
    train.append(fo)
    np.savetxt(f"./{args.file_name}/output/{file_time}/train.txt", train, fmt='%s')
            
            
def toTest(RL_agent, env, eval_env,file_name, args):
    RL_agent.load(f"./{args.file_name}/output/{file_name}/checkpoint/models/")
    evals = []
    start_time = time.time()
    
    for t in range(args.eval):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
    
    
            
def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time,file_time, args, d4rl=False):
    if args.checkpoint or args.test:
        print("---------------------------------------")
        print(f"Evaluation at {t} time steps")
        print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

        total_reward = np.zeros(args.eval_eps)
        for ep in range(args.eval_eps):
            state, done = eval_env.reset(), False
            state = state[0]
            while not done:
                action,_,_,_ = RL_agent.select_action(np.array(state),True)
                next_state, reward, done, _ = eval_env.step(action)
                total_reward[ep] += np.max(reward)
                state = next_state

        print(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f}")
        if d4rl:
            total_reward = eval_env.get_normalized_score(total_reward) * 100
            print(f"D4RL score: {total_reward.mean():.3f}")
        evals.append(total_reward)
        
        if args.checkpoint and not args.test:
            np.save(f"./{args.file_name}/output/{file_time}/checkpoint/{args.file_name}", evals)
            score = np.mean(total_reward) + np.min(total_reward) + np.max(total_reward) + np.median(total_reward) - np.std(total_reward)
            flag = RL_agent.IsCheckpoint(score)
            print(f"This Score：{score} Max Score:{RL_agent.Maxscore}")
            if flag:
                RL_agent.save(f"./{args.file_name}/output/{file_time}/checkpoint/models/")
        if args.test:
            np.save(f"./{args.file_name}/output/{file_time}/evals", evals)
        print("---------------------------------------")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--checkpoint", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=7, type=int)
    parser.add_argument("--max_timesteps", default=200, type=int)
    parser.add_argument("--eval",default=1,type=int)
    # File
    parser.add_argument('--file_name', default=None)
    
    args = parser.parse_args()
    

    
        
    if args.file_name is None:
        args.file_name = f"{args.env}"

    if not os.path.exists(f"./{args.file_name}/output"):
        os.makedirs(f"./{args.file_name}/output")
        
        
    
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(args.env, new_step_api=True)
        eval_env = gym_super_mario_bros.make(args.env, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
        eval_env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
        
    
    env = ProcessEnv(env)
    eval_env = ProcessEnv(eval_env)
    
    

    print("---------------------------------------")
    print(f"Algorithm: PPO, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")





    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    hp = Hyperparameters(
        env=args.env,
        seed=args.seed,
        test=args.test,
        checkpoint=args.checkpoint,
        eval_eps=args.eval_eps,
        max_timesteps=args.max_timesteps,
        eval=args.eval,
        file_name=args.file_name
    )
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    
    envs=SubprocVecEnv(hp.num_processes) if hp.num_processes > 1 else env
    
    
    RL_agent = PPO.agent(state_dim, action_dim, max_action, hp)
    
    rollout = RolloutBuffer(hp.num_steps, hp.num_processes, state_dim, action_dim, hp.gae, hp.discount)
        

    if args.test:
        file_name = f''
        toTest(RL_agent, env, eval_env,file_name, args)
    else:
        from torch.utils.tensorboard import SummaryWriter
        train_online(RL_agent, envs, eval_env, rollout, hp, args)







