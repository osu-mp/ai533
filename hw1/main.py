#!/usr/bin/python3

import numpy as np
import random

# AI 533 - HW1 - Gridworld with value iteration and policy iteration
# Matthew Pacey

# format the output of numpy array printing
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# my debug helper: if true, gridworld values printed at each timestep; if false only final values printed
print_intermediate = False

# maximum number of iterations in each convergence (not really a concern in this scenario)
MAX_ITER = 1000

def get_start_state():
    """
    Return starting state for grid world
    :return:
    """
    util = np.zeros((4, 4))
    util[3, 3] = 100  # goal state, +100 reward
    return util


def get_val(util, x, y):
    """
    Get the value of the requested cell, return None if it is outside the grid
    :param util:
    :param x:
    :param y:
    :return:
    """
    if x < 0 or y < 0 or x > len(util) - 1 or y > len(util[0]) - 1:
        return None
    return util[x, y]


def print_path(util):
    """
    Given a util value matrix, print the direction to follow the highest reward
    :param util:
    :return:
    """
    policy_dir = np.zeros((4, 4), dtype='U1')

    for x in range(len(util)):
        for y in range(len(util[0])):
            value = util[x, y]
            max_value = value

            rt = get_val(util, x, y + 1)
            lt = get_val(util, x, y - 1)
            up = get_val(util, x - 1, y)
            dn = get_val(util, x + 1, y)

            # default behavior: stay in cell if no neighbor has higher val
            policy_dir[x, y] = 'o'
            if rt and rt >= max_value:
                policy_dir[x, y] = '>'
                max_value = rt
            if lt and lt >= max_value:
                policy_dir[x, y] = '<'
                max_value = lt
            if up and up >= max_value:
                policy_dir[x, y] = '^'
                max_value = up
            if dn and dn >= max_value:
                policy_dir[x, y] = 'v'
                max_value = dn

    print(policy_dir)

def get_updated_value(x, y, util, gamma):
    """
    Return the max value for a given state by checking movement in all directions
    :param util:
    :param gamma:
    :return:
    """
    if (x, y) == (3, 3):
        return 100
    if (x, y) == (0, 1) or (x, y) == (0, 2):
        return -10
    if (x, y) == (2, 1) or (x, y) == (2, 2):
        return -5

    # reward of negative 1 for each movement
    cell = -1
    if y < len(util[0]) - 1:
        val_rt = util[x, y + 1]
    else:
        val_rt = cell
    if x > 0:
        val_up = util[x - 1, y]
    else:
        val_up = cell
    if x < len(util) - 1:
        val_dn = util[x + 1, y]
    else:
        val_dn = cell
    if y > 0:
        val_lt = util[x, y - 1]
    else:
        val_lt = cell

    move_rt = 0.8 * (cell + gamma * val_rt) + 0.1 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_dn)
    move_lt = 0.8 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_dn)
    move_up = 0.8 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_rt)
    move_dn = 0.8 * (cell + gamma * val_dn) + 0.1 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_rt)
    return max(move_rt, move_lt, move_up, move_dn)


def value_iteration(max_error, gamma):
    print("\n***************************************************************************")
    print(f"Value iteration for gamma of {gamma} ")

    # initial value of all states
    util = get_start_state()
    # convergence criteria: if all value changes are below this value, exit
    threshold = max_error * (1 - gamma) / gamma

    for t in range(MAX_ITER):
        next_util = np.copy(util)
        # max change of any state in each loop
        max_change = 0

        for x in range(4):
            for y in range(4):
                next_util[x, y] = get_updated_value(x, y, util, gamma)
                change = abs(next_util[x, y] - util[x, y])
                max_change = max(max_change, change)

        if print_intermediate:
            print(f"\nAfter time step {t}")
            print(next_util)
            print(f"{max_change=}, {threshold=}")

        util = next_util

        if max_change <= threshold:
            break

    print(f"\nFinal state for gamma of {gamma} ({max_error=}, total iterations={t})")
    print(util)
    print("***************************************************************************\n")

    return util


def get_random_policy():
    """
    Generate random policy of 4x4 grid with a direction in each cell
    :return:
    """
    policy = np.zeros((4, 4), dtype='U1')
    dirs = ['^', '>', 'v', '<']

    for x in range(len(policy)):
        for y in range(len(policy[0])):
            policy[x, y] = dirs[random.randint(0, len(dirs) - 1)]

    return policy


def are_policies_the_same(policy_1, policy_2):
    """
    Returns true iff the two policies are the same
    :param policy_1:
    :param policy_2:
    :return:
    """
    for x in range(len(policy_1)):
        for y in range(len(policy_1[0])):
            if policy_1[x, y] != policy_2[x, y]:
                return False

    return True


def get_updated_value_from_policy(x, y, util, gamma, policy):
    """
    Given a policy, compute the util value for the given x,y cell
    """
    """
    :param util:
    :param gamma:
    :return:
    """
    if (x, y) == (3, 3):
        return 100
    if (x, y) == (0, 1) or (x, y) == (0, 2):
        return -10
    if (x, y) == (2, 1) or (x, y) == (2, 2):
        return -5

    dir = policy[x, y]

    cell = -1 # util[x, y]
    if y < len(util[0]) - 1:
        val_rt = util[x, y + 1]
    else:
        val_rt = cell
    if x > 0:
        val_up = util[x - 1, y]
    else:
        val_up = cell
    if x < len(util) - 1:
        val_dn = util[x + 1, y]
    else:
        val_dn = cell
    if y > 0:
        val_lt = util[x, y - 1]
    else:
        val_lt = cell

    move_rt = 0.8 * (cell + gamma * val_rt) + 0.1 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_dn)
    move_lt = 0.8 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_dn)
    move_up = 0.8 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_rt)
    move_dn = 0.8 * (cell + gamma * val_dn) + 0.1 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_rt)

    if dir == '>':
        return move_rt
    elif dir == 'v':
        return move_dn
    elif dir == '<':
        return move_lt
    elif dir == '^':
        return move_up
    else:
        raise Exception(f'Policy gave garbage direction: {dir}')


def get_dir_of_max_neighbor(x, y, util):
    """
    Given a cell, get the direction of the neighbor with the highest util
    :param x:
    :param y:
    :param util:
    :return:
    """
    max = -100

    val_rt = get_val(util, x, y + 1)
    val_lt = get_val(util, x, y - 1)
    val_up = get_val(util, x - 1, y)
    val_dn = get_val(util, x + 1, y)

    if val_rt and val_rt > max:
        dir = '>'
        max = val_rt
    if val_up and val_up > max:
        dir = '^'
        max = val_up
    if val_dn and val_dn > max:
        dir = 'v'
        max = val_dn
    if val_lt and val_lt > max:
        dir = '<'
        max = val_lt

    return dir


def policy_iteration(max_error, gamma):
    """
    Policy iteration
    :param max_error:
    :param gamma:
    :return:
    """
    print("\n***************************************************************************")
    print(f"Policy iteration for gamma of {gamma} ({max_error=})")

    util = get_start_state()

    # init random policy
    policy = get_random_policy()
    # threshold for considering value matrices equivalent
    threshold = max_error * (1 - gamma) / gamma
    # number of iterations in value convergence
    iter_cnt = 0

    # repeat outer loop until no change in policy
    while True:

        # repeat this inner loop until the util values converge
        while True:
            next_util = np.copy(util)
            max_change = 0

            for x in range(4):
                for y in range(4):
                    # update the util value using the current policy (not best move like VI)
                    next_util[x, y] = get_updated_value_from_policy(x, y, util, gamma, policy)
                    change = abs(next_util[x, y] - util[x, y])
                    max_change = max(max_change, change)
            iter_cnt += 1

            # debug
            if print_intermediate:
                print(next_util)
                print(f"{iter_cnt=}, {max_change=}, {threshold=}")

            util = next_util
            # break out when util val converges
            if max_change <= threshold:
                break

        # for each state, update the policy to move to the state with the highest reward
        new_policy = np.copy(policy)
        for x in range(4):
            for y in range(4):
                new_policy[x, y] = get_dir_of_max_neighbor(x, y, next_util)

        if print_intermediate:
            print(new_policy)
            print(f"{max_change=}, {threshold=}")

        # if there was no change in the policy, we can exit the outer loop
        if are_policies_the_same(policy, new_policy):
            break

        policy = np.copy(new_policy)

    print(f"\nFinal state for gamma of {gamma} ({max_error=}, total iterations={iter_cnt})")
    print(util)
    print("***************************************************************************\n")

    return util, policy

def main():
    """
    Main func: run different flavors of VI and PI (will be compared in report)
    :return:
    """
    max_error = 0.0001
    util = value_iteration(max_error=max_error, gamma=0.3)
    print_path(util)
    util = value_iteration(max_error=max_error, gamma=0.95)
    print_path(util)
    util, policy = policy_iteration(max_error=max_error, gamma=0.95)
    print_path(util)
    print("DONE")


if __name__ == '__main__':
    main()
