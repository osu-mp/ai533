#!/usr/bin/python3

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

from actor_critic import Agent

"""
AI 533 - HW4 - SARSA-lambda and Q-learning-lamda
Matthew Pacey
README: just run this script directly, it will pick the hyperparams 
and generate a plot of the performance of the two algorithms in the gridworld example
"""
# format the output of numpy array printing
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# my debug helper: if true, gridworld values printed at each timestep; if false only final values printed
DEBUG = False

gamma = 0.95                # future reward discount
NUM_EPISODES = 100          # episode count in each trial
NUM_TRIALS = 100            # number of trials to run
MAX_STEPS = 1000            # stop episodes that do not reach goal before this many steps

actions = ['^', '>', 'v', '<']

"""
This is the policy for the given Gridworld. Starting at the top (0,0) or s0, the ideal
path will be to go down along the left side to the bottom and then right to the goal (3,3) or s15.
"""
policy = np.zeros((4, 4), dtype='U1')
policy = [
    ['v', 'v', '>', 'v'],
    ['v', '>', '>', 'v'],
    ['v', 'v', 'v', 'v'],
    ['>', '>', '>', 'o'],
]

rewards = [
    [-1, -10, -10, -1],
    [-1, -1, -1, -1],
    [-1, -5, -5, -1],
    [-1, -1, -1, 100],
]


def get_init_q():
    """
    Init all state-action pairs to a nominal value
    :return:
    """
    q = {}
    for x in range(4):
        for y in range(4):
            for a in actions:
                q[((x, y), a)] = 0

    return q


def get_move_dir(action):
    """
    Given a desired action, probabilistically determine the actual move.
    The transition function is 80% chance of going in desired direction, with a 10%
    chance of sliding in neighboring direction. For example, if the desired action is up,
    there is an 80% chance of going up, 10% left and 10% right (0% down).
    :param action:
    :return:
    """
    if action == '>':       # move right, can slide up or down
        moves = ['>', '^', 'v']
    elif action == 'v':     # move down, can slide left or right
        moves = ['v', '<', '>']
    elif action == '<':     # move left, can slide up or down
        moves = ['<', '^', 'v']
    if action == '^':       # move up, can slide left or right
        moves = ['^', '<', '>']

    # get a number between 0 and 1
    # if it is less than 0.8, move in primary direction
    # else if less than 0.9, slide in one direction
    # else, slide in other direction
    prob = random.random()
    if prob <= 0.8:
        move = moves[0]
    elif prob <= 0.9:
        move = moves[1]
    else:
        move = moves[2]

    return move


def move_action(s, action):
    """
    Given starting x,y and action, perform action
    Ensure the agent stays on the grid
    :param x:
    :param y:
    :param action:
    :return:
    """
    x, y = s
    if action == '>':
        y = min(y + 1, 3)
    elif action == '<':
        y = max(y - 1, 0)
    elif action == '^':
        x = max(x - 1, 0)
    elif action == 'v':
        x = min(x + 1, 3)
    else:
        raise Exception(f"Invalid action {action}")

    return x, y


def get_epsilon_greedy_action(q, s, epsilon):
    """
    Give q-values and state, determine action using epsilon-greedy policy
    :param q:
    :param s:
    :return:
    """
    # determine whether to use best action or random
    ep_val = random.random()
    if ep_val <= epsilon:           # choose random action
        desired_action = random.choice(actions)
    else:
        desired_action = get_max_q_value(q, s)

    actual_action = get_move_dir(desired_action)

    if DEBUG:
        print(f"{ep_val=}, {desired_action=}, {actual_action=}")

    return actual_action#, max_value


def get_max_q_value(q, s, a_prime=None):
    """
    Given a particular state, return the max q value over all actions
    :param s:
    :return:
    """
    max_value = float('-inf')
    poss_actions = []
    for action in actions:
        value = q[(s, action)]
        if value > max_value:
            poss_actions = [action]
            max_value = value
        elif value == max_value:
            poss_actions.append(action)

    if a_prime and a_prime in poss_actions:
        return a_prime

    action = random.choice(poss_actions)
    return action


def SARSA_lambda(alpha, epsilon, lam):
    """
    Run the SARSA-lambda (state action reward, state action) algo
    with eligibility traces (backward view, accumulating)
    :return:
    """
    all_trials = []

    for trial in range(NUM_TRIALS):
        trial_rewards = []
        q = get_init_q()                    # q-values
        e = get_init_q()                    # eligibility traces

        for ep in range(NUM_EPISODES):
            s = (0, 0)
            a = policy[s[0]][s[1]]

            reward = 0
            for step in range(MAX_STEPS):
                if DEBUG:
                    print(f"{step}: {s=}")
                # take action a, observe R, s'
                s_prime = move_action(s, a)
                r = rewards[s_prime[0]][s_prime[1]]

                # choose a' from s' using epsilon greedy policy
                a_prime = get_epsilon_greedy_action(q, s_prime, epsilon)
                reward += r
                delta = r + gamma * q[(s_prime, a_prime)] - q[(s, a)]
                # print(f"{s=}, {a=}, {delta=}, {gamma=}")
                # print(f"e before:\n{e}")
                e[s, a] += 1
                # print(f"e after:\n{e}")

                # print(f"q before:\n{q}")
                for x in range(4):
                    for y in range(4):
                        for act in actions:
                            q[((x, y), act)] += alpha * delta * e[((x, y), act)]
                            e[((x, y), act)] *= gamma * lam
                # print(f"q after:\n{q}")
                # print(f"e after update:\n{q}")
                s = s_prime
                a = a_prime
                if s == (3, 3):
                    if DEBUG:
                        print(f"GOAL ({step} steps)")
                    trial_rewards.append(reward)
                    break

        all_trials.append(trial_rewards)
    return all_trials

def q_learning_lambda(alpha, epsilon, lam):
    """
    Run the q-learning-lambda algo in gridworld
    with eligibility traces (backward view, accumulating)
    :param alpha:
    :param epsilon:
    :return:
    """
    all_trials = []

    for trial in range(NUM_TRIALS):
        trial_rewards = []
        q = get_init_q()
        e = get_init_q()

        for ep in range(NUM_EPISODES):
            s = (0, 0)
            a = policy[s[0]][s[1]]
            reward = 0
            for step in range(MAX_STEPS):
                if DEBUG:
                    print(f"{step}: {s=}")
                # take action a, observe R, s'
                s_prime = move_action(s, a)
                r = rewards[s_prime[0]][s_prime[1]]

                a_prime = get_epsilon_greedy_action(q, s_prime, epsilon)
                reward += r

                # get max reward from s_prime
                a_star = get_max_q_value(q, s_prime, a_prime)
                delta = r + gamma * q[(s_prime, a_star)] - q[s, a]
                e[s, a] += 1

                for x in range(4):
                    for y in range(4):
                        for act in actions:
                            q[((x, y), act)] += alpha * delta * e[((x, y), act)]
                            if a_prime == a_star:
                                e[((x, y), act)] *= gamma * lam
                            else:
                                e[((x, y), act)] = 0

                s = s_prime
                a = a_prime
                if s == (3, 3):
                    if DEBUG:
                        print(f"GOAL ({step} steps)")
                    trial_rewards.append(reward)
                    break

        all_trials.append(trial_rewards)

    return all_trials


def plot_trials(sarsa, ql, alpha, epsilon, lam):
    plt.clf()
    plt.title(f"Average Rewards ($\\alpha$={alpha}, $\epsilon$={epsilon}, $\\lambda$={lam})")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward (with Error Bars)")

    sarsa = np.array(sarsa)
    ql = np.array(ql)

    sarsa_y, sarsa_err = [], []
    ql_y, ql_err = [], []
    for ep in range(NUM_EPISODES):
        sarsa_y.append(np.average(sarsa[:,ep]))
        sarsa_err.append(np.std(sarsa[:,ep]))

        ql_y.append(np.average(ql[:, ep]))
        ql_err.append(np.std(ql[:, ep]))

    # make the bottom line wider since both plots overlap a lot
    plt.errorbar(list(range(NUM_EPISODES)), ql_y, ql_err, label="Q-Learning", color='blue', elinewidth=2.2)
    plt.errorbar(list(range(NUM_EPISODES)), sarsa_y, sarsa_err, label="SARSA", color='red', elinewidth=1.1)

    plt.legend(loc='lower right', title='Algorithm')

    fname = "hw4-paceym-results.png"
    plt.savefig(fname, dpi=250)
    print(f"Plot saved to {fname}")


def tune_hyperparams():
    """
    Rough way to estimate hyperparams:
    Take average of all trials/episodes across both algos and use the combo with the highest average
    :return:
    """
    best_alpha, best_epsilon, best_lambda = 0, 0, 0
    max_value = float('-inf')
    print("Determining optimal hyperparams")

    for alpha in [0.05, 0.17, 0.34]: # [0.0001, 0.01]:#, 0.1, 0.2, 0.35, 0.5]:
        for epsilon in [0.0001, 0.1, 0.2]:#, 0.1, 0.2, 0.35, 0.5]:
            for lam in [0, 0.05, 0.1, 0.25, 0.5, 0.7, 0.9999]:
                sarsa = np.average(SARSA_lambda(alpha, epsilon, lam))
                ql = np.average(q_learning_lambda(alpha, epsilon, lam))
                average = np.average([sarsa, ql])

                print(f"{alpha=:0.4f}, {epsilon=:0.4f}, {lam=:0.4f} : Avg. reward {average:2.4f}")
                if average > max_value:
                    max_value = average
                    best_alpha, best_epsilon, best_lambda = alpha, epsilon, lam


    print(f"\nHighest average: {best_alpha=},{best_epsilon=}, {best_lambda=},{max_value}\n")
    return best_alpha, best_epsilon, best_lambda


def actor_critic():
    env = gym.make("CartPole-v0")
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1800

    filename = 'cartpole.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f"episode {i} score {score:%.1f} {avg_score:%.1f}")

    x = [i+1 for l in range(n_games)]
    # TODO
    # plot_learning_curve(x, score_history, figure_file)

def main():
    # SARSA and Q-Learning
    # alpha, epsilon, lam = tune_hyperparams()
    #
    # sarsa_rewards = SARSA_lambda(alpha, epsilon, lam)
    # ql_rewards = q_learning_lambda(alpha, epsilon, lam)
    # plot_trials(sarsa_rewards, ql_rewards, alpha, epsilon, lam)

    # Actor Critic
    actor_critic()



if __name__ == '__main__':
    main()
