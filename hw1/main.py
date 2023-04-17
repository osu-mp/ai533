#!/usr/bin/python3

import numpy as np

# AI 533 - HW1 - Gridworld
# Matthew Pacey

# TODO : no penalty for each movement (YET)

# set
np.set_printoptions(precision=4)

# if true, gridworld values printed at each timestep; if false only final values printed
print_intermediate = False

def get_start_state():
    """
    Return starting state for grid world
    :return:
    """
    util = np.zeros((4, 4))
    util[3, 3] = 100  # goal state, +100 reward
    # util[0, 1] = -10  # fire, -10 reward
    # util[0, 2] = -10  # fire, -10 reward
    # util[2, 1] = -5  # water, -5 reward
    # util[2, 2] = -5  # water, -5 reward
    return util

def get_val(util, x, y):
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
            # print(f"{rt=}, {lt=}, {up=}, {dn=}")

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

    # TODO : what happens if the agent cannot move (i.e. edge cell)
    #   does that mean the value is 0 or the value of the current cell?

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

    if x == 0 and y == 2:
        pass

    move_rt = 0.8 * (cell + gamma * val_rt) + 0.1 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_dn)
    move_lt = 0.8 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_dn)
    move_up = 0.8 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_rt)
    move_dn = 0.8 * (cell + gamma * val_dn) + 0.1 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_rt)
    return max(move_rt, move_lt, move_up, move_dn)

def value_iteration(mdp, max_error, gamma):
    print("\n***************************************************************************")
    print(f"Value iteration for gamma of {gamma} ({max_error=})")
    # initial value of all states
    util = get_start_state()
    # max change of any state in each loop

    print(f"init state")
    print(util)

    for t in range(500):
        next_util = np.copy(util)
        max_change = 0

        if t == 2:
            pass

        for x in range(4):
            for y in range(4):
                # print(f"{next_util[x, y]}")
                # move right only now
                # next_util[x, y] = 0.8*(util[x, y] + gamma * util[x, y + 1]) # TODO add 0.1 and 0.1 for other
                next_util[x, y] = get_updated_value(x, y, util, gamma)
                change = abs(next_util[x, y] - util[x, y])
                max_change = max(max_change, change)

        # next_util[0, 2] = 0.72
        # for idx, x in np.ndenumerate(next_util):
        #     next_util[idx] = 0.8*(x + gamma * )
        #     pass
        if print_intermediate:
            print(f"\nAfter time step {t}")
            print(next_util)
            print(f"{max_change=}, {threshold=}")

        util = next_util
        threshold = max_error * (1 - gamma) / gamma
        if max_change <= threshold:
            break

    print(f"\nFinal state for gamma of {gamma} ({max_error=}, total iterations={t}")
    print(util)
    print("***************************************************************************\n")

    return util


def main():
    util = value_iteration(None, max_error=0.001, gamma=0.3)
    print_path(util)
    util = value_iteration(None, max_error=0.001, gamma=0.95)
    print_path(util)
    print("DONE")


if __name__ == '__main__':
    main()
