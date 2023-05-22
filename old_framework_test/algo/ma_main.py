import sys
sys.path.append('F:\githubClone\Multi_agent_AAC\old_framework_test')
from env.make_env import make_env
import argparse, datetime
from tensorboardX import SummaryWriter
import numpy as np
import torch, os
import wandb
from algo.maddpg.maddpg_agent import MADDPG
from algo.normalized_env import ActionNormalizedEnv, ObsEnv, reward_from_state
from algo.utils import *
from copy import deepcopy


def main(args):
    # simulation result saving
    today = datetime.date.today()
    current_date = today.strftime("%d%m%y")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H_%M_%S")
    # wandb.login(key="efb76db851374f93228250eda60639c70a93d1ec")
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="MADDPG_sample_newFrameWork",
    #     name='MADDPG_test_'+str(current_date) + '_' + str(formatted_time),
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": args.a_lr,
    #         "epochs": args.max_episodes,
    #     }
    # )

    env = make_env(args.scenario)
    n_agents = env.n
    n_actions = env.world.dim_p
    # env = ActionNormalizedEnv(env)
    # env = ObsEnv(env)
    n_states = env.observation_space[0].shape[0]

    torch.manual_seed(args.seed)

    # if args.tensorboard and args.mode == "train":
    #     writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.log_dir)

    if args.algo == "maddpg":
        model = MADDPG(n_states, n_actions, n_agents, args)

    print(model)
    #model.load_model()

    episode = 0
    total_step = 0

    while episode < args.max_episodes:

        state = env.reset()

        episode += 1
        step = 0
        accum_reward = 0
        rewardA = 0
        rewardB = 0
        rewardC = 0
        while True:

            if args.mode == "train":
                action = model.choose_action(state, noisy=True)
                next_state, reward, done, info = env.step(action)

                step += 1
                total_step += 1
                reward = np.array(reward)

                rew1 = reward_from_state(next_state)
                reward = rew1 + (np.array(reward, dtype=np.float32) / 100.)
                accum_reward += sum(reward)
                rewardA += reward[0]
                rewardB += reward[1]
                rewardC += reward[2]


                if args.algo == "maddpg" or args.algo == "commnet":
                    obs = torch.from_numpy(np.stack(state)).float().to(device)
                    obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)
                    if step != args.episode_length - 1:
                        next_obs = obs_
                    else:
                        next_obs = None
                    rw_tensor = torch.FloatTensor(reward).to(device)
                    ac_tensor = torch.FloatTensor(action).to(device)
                    if args.algo == "commnet" and next_obs is not None:
                        model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor)
                    if args.algo == "maddpg":
                        model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor)
                    obs = next_obs
                else:
                    model.memory(state, action, reward, next_state, done)

                state = next_state

                if args.episode_length < step or (True in done):
                    # c_loss, a_loss = model.update(episode)
                    c_loss, a_loss = model.update_myown(episode)

                    print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                    # wandb.log({'overall_reward': float(accum_reward)})
                    if c_loss and a_loss:
                        for idx, val in enumerate(c_loss):
                            print(" agent %s, a_loss %3.2f c_loss %3.2f" % (idx, a_loss[idx].item(), c_loss[idx].item()))
                            # wandb.log({'agent' + str(idx) + 'actor_loss': float(a_loss[idx].item())})
                            # wandb.log({'agent' + str(idx) + 'critic_loss': float(c_loss[idx].item())})
                            print(" agent %s, a_loss %3.2f c_loss %3.2f" % (idx, a_loss[idx].item(), c_loss[idx].item()))
                            # wandb.log({'agent' + str(idx) + 'actor_loss': float(a_loss[idx].item())})
                            # wandb.log({'agent' + str(idx) + 'critic_loss': float(c_loss[idx].item())})
                            print(" agent %s, a_loss %3.2f c_loss %3.2f" % (idx, a_loss[idx].item(), c_loss[idx].item()))
                            # wandb.log({'agent' + str(idx) + 'actor_loss': float(a_loss[idx].item())})
                            # wandb.log({'agent' + str(idx) + 'critic_loss': float(c_loss[idx].item())})

                    if episode % args.save_interval == 0 and args.mode == "train":
                        model.save_model(episode)

                    env.reset()
                    # model.reset()
                    break
            elif args.mode == "eval":
                action = model.choose_action(state, noisy=False)
                next_state, reward, done, info = env.step(action)
                step += 1
                total_step += 1
                state = next_state
                reward = np.array(reward)
                import time
                time.sleep(0.02)
                env.render()

                rew1 = reward_from_state(next_state)
                reward = rew1 + (np.array(reward, dtype=np.float32) / 100.)
                accum_reward += sum(reward)
                rewardA += reward[0]
                rewardB += reward[1]
                rewardC += reward[2]

                if args.episode_length < step or (True in done):
                    print("[Episode %05d] reward %6.4f " % (episode, accum_reward))
                    env.reset()
                    break
    # wandb.finish()

    # if args.tensorboard:
    #     writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="simple_spread", type=str)
    parser.add_argument('--max_episodes', default=60000, type=int)  # rnu for a total of 60000 episodes
    parser.add_argument('--algo', default="maddpg", type=str, help="commnet/bicnet/maddpg")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--episode_length', default=50, type=int)
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=5000, type=int)  # save model for every 5000 episodes
    parser.add_argument("--model_episode", default=240000, type=int)
    parser.add_argument('--episode_before_train', default=1000, type=int)  # original 1000
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    args = parser.parse_args()

    main(args)
