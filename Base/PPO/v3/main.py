import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import gym_super_mario_bros.actions
import argparse
import os
import time
import datetime
import PPO
from pre_env import ProcessEnv




"""
DQN或者AC的方法应用到这里，都有个问题：
他们学习的是全局的策略，而不是具体目标的应对策略
也就是说，他们学到的是在哪个位置怎么做，而不是靠近某个目标怎么做

256批量，一次训练0.05s

由离线强化学习看到，online的rl肯定会在原来的replaybuffer上更进一步的，否则它怎么不断迭代找到最优解？
所有在离线强化学习中比当时的data展示的policy好，是必然的，rl可以在reward信号引导下，分清好坏
                


"""





def train_online(RL_agent, env, args):
    def handle_episode_finish(ep_num, t, rounds, ep_timesteps, ep_total_reward):
        print(f"T: {t} Total T: {rounds+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")
        writer.add_scalar('round_reward', ep_total_reward, global_step=rounds)
        state, done = env.reset(), False
        state = state[0]
        hidden = (np.zeros((512), dtype=np.float32), np.zeros((512), dtype=np.float32))
        return state, 0, 0, ep_num + 1, False, hidden

    state, ep_finished = env.reset(), False
    state = state[0]
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1
    rounds = 0
    hidden = (np.zeros((512), dtype=np.float32), np.zeros((512), dtype=np.float32))
    start_time = time.time()

    writer = SummaryWriter(f'./{args.file_name}/results/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}-PPO-{args.max_timesteps}')

    for t in range(int(args.max_timesteps+1)):
        flag = False
        while True:
            rounds += 1
            action, logit, value, hidden = RL_agent.select_action(np.array(state),hidden)
            next_state, reward, ep_finished, _, _ = env.step(action)

            ep_total_reward += reward
            ep_timesteps += 1

            done = float(ep_finished)

            if flag:
                if ep_finished:
                    state, ep_total_reward, ep_timesteps, ep_num, done, hidden = handle_episode_finish(ep_num, t, rounds, ep_timesteps, ep_total_reward)
                next_value = RL_agent.get_value(np.array(next_state),hidden)
                RL_agent.replaybuffer.computeReturn(next_value, done)
                print(f"T: {t} Total T: {rounds+1}  begintrain！！")
                actor_loss, value_loss = RL_agent.train()
                writer.add_scalar('actor_loss', actor_loss, global_step=rounds)
                writer.add_scalar('value_loss', value_loss, global_step=rounds)
                RL_agent.replaybuffer.flagtoFalse()
                evals = []
                if args.checkpoint:
                    maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
                break

            flag = RL_agent.replaybuffer.add(state, action, value, reward, done, logit, hidden)

            state = next_state
            
            if ep_finished:
                state, ep_total_reward, ep_timesteps, ep_num, done, hidden = handle_episode_finish(ep_num, t, rounds, ep_timesteps, ep_total_reward)
            
            
def toTest(RL_agent, env, eval_env, args):
    RL_agent.load(f"./{args.file_name}/checkpoint/models/")
    evals = []
    start_time = time.time()
    
    for t in range(args.eval):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
    
    

            
def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=False):
    if args.checkpoint or args.test:
        print("---------------------------------------")
        print(f"Evaluation at {t} time steps")
        print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

        total_reward = np.zeros(args.eval_eps)
        for ep in range(args.eval_eps):
            state, done = eval_env.reset(), False
            state = state[0]
            hidden = (np.zeros((512), dtype=np.float32), np.zeros((512), dtype=np.float32))
            while not done:
                action,_,_,_ = RL_agent.select_action(np.array(state),hidden)
                if args.test:
                    action = RL_agent.select_action(np.array(state),hidden)
                next_state, reward, done, _, _ = eval_env.step(action)
                total_reward[ep] += reward
                state = next_state

        print(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f}")
        if d4rl:
            total_reward = eval_env.get_normalized_score(total_reward) * 100
            print(f"D4RL score: {total_reward.mean():.3f}")
        evals.append(total_reward)
        
        if args.checkpoint and not args.test:
            np.save(f"./checkpoint/{args.file_name}", evals)
            score = np.mean(total_reward) + np.min(total_reward) + np.max(total_reward) + np.median(total_reward) - np.std(total_reward)
            flag = RL_agent.IsCheckpoint(score)
            print(f"This Score：{score} Max Score:{RL_agent.Maxscore}")
            if flag:
                RL_agent.save(f"./{args.file_name}/checkpoint/models/")
        if args.test:
            np.save(f"./{args.file_name}/results/{args.file_name}", evals)
        print("---------------------------------------")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="SuperMarioBros-v0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--checkpoint", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=5, type=int)
    parser.add_argument("--max_timesteps", default=200, type=int)
    parser.add_argument("--eval",default=1,type=int)
    # File
    parser.add_argument('--file_name', default=None)
    
    args = parser.parse_args()
    
    
    
    
    if args.file_name is None:
        args.file_name = f"{args.env}"

    if not os.path.exists(f"./{args.file_name}/results"):
        os.makedirs(f"./{args.file_name}/results")
        
    if not os.path.exists(f"./{args.file_name}/models"):
        os.makedirs(f"./models")
        
    if args.checkpoint and not os.path.exists(f"./{args.file_name}/checkpoint"):
        os.makedirs(f"./{args.file_name}/checkpoint/models")
        
    
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(args.env, new_step_api=True)
        eval_env = gym_super_mario_bros.make(args.env, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(args.env, apply_api_compatibility=True)
        eval_env = gym_super_mario_bros.make(args.env,render_mode = 'human', apply_api_compatibility=True)
        
        
    env = ProcessEnv(env)
    eval_env = ProcessEnv(eval_env)
    
    

    print("---------------------------------------")
    print(f"Algorithm: PPO, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    RL_agent = PPO.agent(state_dim, action_dim, args.test)
        
        

    if args.test:
        toTest(RL_agent, env, eval_env, args)
    else:
        from torch.utils.tensorboard import SummaryWriter
        train_online(RL_agent, env, args)







