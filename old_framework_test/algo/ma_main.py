import sys
# sys.path.append('F:\githubClone\Multi_agent_AAC\old_framework_test')
sys.path.append('D:\Multi_agent_AAC\old_framework_test')
from env.make_env import make_env
import argparse, datetime
import numpy as np
import torch, os
import wandb
from parameters_V5_2 import initialize_parameters
from algo.maddpg.maddpg_agent import MADDPG
from algo.normalized_env import ActionNormalizedEnv, ObsEnv, reward_from_state
from algo.utils import *
from copy import deepcopy
import torch
import csv


def main(args):
    num_devices = torch.cuda.device_count()
    print("Number of GPUs:", num_devices)
    # Get the names of the available GPUs
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_devices)]
    print("GPU Names:", gpu_names)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    today = datetime.date.today()
    current_date = today.strftime("%d%m%y")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H_%M_%S")
    file_name = 'D:\MADDPG_2nd_jp/' + str(current_date) + '_' + str(formatted_time)
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    plot_file_name = file_name + '/toplot'
    if not os.path.exists(plot_file_name):
        os.makedirs(plot_file_name)

    wandb.login(key="efb76db851374f93228250eda60639c70a93d1ec")
    wandb.init(
        # set the wandb project where this run will be logged
        project="MADDPG_sample_newFrameWork",
        name='MADDPG_test_'+str(current_date) + '_' + str(formatted_time),
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.a_lr,
            "epochs": args.max_episodes,
        }
    )

    # env = make_env(args.scenario)  # original environment

    # -------------- create my own environment -----------------
    n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, \
    agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY, seed_used, max_xy = initialize_parameters()
    total_agentNum = 2
    max_nei_num = 5
    # create world
    actor_obs = [6+(total_agentNum-1)*2, 20, 6]  # dim host, maximum dim grid, dim other drones
    critic_obs = [6+(total_agentNum-1)*2, 20, 6]
    n_actions = 2

    # original
    # actorNet_lr = learning_rate
    # criticNet_lr = learning_rate

    actorNet_lr = 1e-4
    criticNet_lr = 1e-3

    # noise parameter ini
    largest_Nsigma = 0.5
    smallest_Nsigma = 0.15
    ini_Nsigma = largest_Nsigma

    max_spd = 15
    env.create_world(total_agentNum, n_actions, GAMMA, TAU, UPDATE_EVERY, largest_Nsigma, smallest_Nsigma, ini_Nsigma, max_xy, max_spd)
    #


    # --------- original -----------
    # n_agents = env.n
    # n_actions = env.world.dim_p
    # n_states = env.observation_space[0].shape[0]

    # --------- my own -----------
    n_agents = len(env.all_agents)
    n_actions = n_actions
    n_states = actor_obs[0]


    torch.manual_seed(args.seed)  # this is the seed

    # if args.tensorboard and args.mode == "train":
    #     writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.log_dir)

    if args.algo == "maddpg":
        model = MADDPG(n_states, n_actions, n_agents, args, criticNet_lr, actorNet_lr, GAMMA, TAU)

    print(model)
    #model.load_model()

    episode = 0
    total_step = 0
    score_history = []

    while episode < args.max_episodes:

        # state = env.reset()  # original env reset

        # ------------ my own env.reset() ------------ #
        cur_state, norm_cur_state = env.reset_world(show=0)


        episode += 1
        step = 0
        accum_reward = 0
        rewardA = 0
        rewardB = 0
        rewardC = 0
        while True:

            if args.mode == "train":
                action = model.choose_action(norm_cur_state, noisy=True)
                next_state, norm_next_state = env.step(action, step)
                reward_aft_action, done_aft_action, check_goal = env.get_step_reward(step)


                step += 1  # current play step
                total_step += 1  # steps taken from 1st episode

                if args.algo == "maddpg" or args.algo == "commnet":
                    # obs = torch.from_numpy(np.stack(cur_state)).float().to(device)
                    obs = torch.from_numpy(np.stack(norm_cur_state)).float().to(device)
                    # obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)
                    obs_ = torch.from_numpy(np.stack(norm_next_state)).float().to(device)
                    next_obs = obs_

                    rw_tensor = torch.FloatTensor(np.array(reward_aft_action)).to(device)
                    ac_tensor = torch.FloatTensor(action).to(device)
                    done_tensor = torch.FloatTensor(done_aft_action).to(device)
                    if args.algo == "commnet" and next_obs is not None:
                        model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor, done_tensor)
                    if args.algo == "maddpg":
                        model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor, done_tensor)
                    obs = next_obs
                else:
                    model.memory(cur_state, action, reward_aft_action, next_state, done_aft_action)

                accum_reward = accum_reward + sum(reward_aft_action)

                c_loss, a_loss = model.update_myown(episode, total_step,
                                                    UPDATE_EVERY)  # last working learning framework

                cur_state = next_state
                norm_cur_state = norm_next_state


                if args.episode_length < step or (True in done_aft_action):
                    # here onwards is end of an episode's play
                    score_history.append(accum_reward)
                    print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                    wandb.log({'overall_reward': float(accum_reward)})
                    if c_loss and a_loss:
                        for idx, val in enumerate(c_loss):
                            print(" agent %s, a_loss %3.2f c_loss %3.2f" % (
                                idx, a_loss[idx].item(), c_loss[idx].item()))
                            wandb.log({'agent' + str(idx) + 'actor_loss': float(a_loss[idx].item())})
                            wandb.log({'agent' + str(idx) + 'critic_loss': float(c_loss[idx].item())})
                    if episode % args.save_interval == 0 and args.mode == "train":

                        # save the models at a predefined interval
                        # save model to my own directory
                        filepath = file_name+'/interval_record_eps'
                        model.save_model(episode, filepath)  # this is the original save model

                    # cur_state, norm_cur_state = env.reset_world(show=0)
                    # model.reset()
                    break  # this is to break out from "while True:", which is one play
            elif args.mode == "eval":
                action = model.choose_action(cur_state, noisy=False)
                next_state, reward, done, info = env.step(action)
                step += 1
                total_step += 1
                cur_state = next_state
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
    wandb.finish()

    # if args.tensorboard:
    #     writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="simple_spread", type=str)
    parser.add_argument('--max_episodes', default=30000, type=int)  # rnu for a total of 60000 episodes
    parser.add_argument('--algo', default="maddpg", type=str, help="commnet/bicnet/maddpg")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--episode_length', default=50, type=int)  # maximum play per episode
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)  # original 256
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=1000, type=int)  # save model for every 5000 episodes
    parser.add_argument("--model_episode", default=60000, type=int)
    parser.add_argument('--episode_before_train', default=10, type=int)  # original 1000
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    args = parser.parse_args()

    main(args)
