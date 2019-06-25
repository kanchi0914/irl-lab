import sys
from contextlib import closing

import numpy as np
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

import  random

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],

    "4x4_2": [
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG"
    ],
    # "8x8": [
    #     "SFFFFFFF",
    #     "FFFFFFFF",
    #     "FFFHFFFF",
    #     "FFFFFHFF",
    #     "FFFHFFFF",
    #     "FHHFFFHF",
    #     "FHFFHFHF",
    #     "FFFHFFFG"
    # ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "GFFFFFFG"
    ],

    "DST_10*11": [
        "SFFFFFFFFF",
        "GFFFFFFFFF",
        "WGFFFFFFFF",
        "WWGFFFFFFF",
        "WWWGGGFFFF",
        "WWWWWWFFFF",
        "WWWWWWFFFF",
        "WWWWWWGGFF",
        "WWWWWWWWFF",
        "WWWWWWWWGF",
        "WWWWWWWWWG"
    ],

    "test": [
        "SFFFF",
        "GFFFF",
        "WGFFF",
        "WWGFF",
        "WWWWG"
    ],

    "cliff": [
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "SHHHHHHHHHHG",
    ],

    "cliff2": [
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "SFFFFFFFFFFG",
    ],

}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] not in '#H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class DeepSeaEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]

        self.map_name = map_name
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        self.shape = desc.shape

        nA = 4

        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def from_s(n):
            return

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                letter = desc[row, col]

                w1 = 0.0
                r_st = -1

                if (letter == b'W'):
                    for a in range(4):
                        li = P[s][a]
                        li.append((1.0, s, r_st, True))
                else:
                    for a in range(4):
                        li = P[s][a]
                        if letter in b'GH':
                            li.append((1.0, s, r_st, True))
                        else:

                            if is_slippery:
                                pass
                                # for b in [(a-1)%4, a, (a+1)%4]:
                                #     newrow, newcol = inc(row, col, b)
                                #     newstate = to_s(newrow, newcol)
                                #     newletter = desc[newrow, newcol]
                                #     done = bytes(newletter) in b'GH'
                                #     rew = float(newletter == b'G')
                                #     li.append((1.0/3.0, newstate, rew, done))

                            else:

                                newrow, newcol = inc(row, col, a)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]

                                #print("action:", a, " old:", row, ",", col, " new:", newrow, ",", newcol)
                                #print("action:", a, " old:", state0, " new:", newstate)

                                #壁にぶつかった
                                if (newletter == b'W'):
                                    li.append((1.0, s, r_st, False))
                                else:
                                    # done = bytes(newletter) in b'GH'

                                    if (self.map_name == "cliff"):

                                        r_st = -1

                                        per = 0.00
                                        r_goal = 100
                                        r_drop = -100

                                        if (newletter == b'G'):
                                            t_reward = r_st + r_goal
                                            li.append((1.0, newstate, t_reward, True))

                                        elif (newletter == b'H'):
                                            t_reward = r_st + r_drop
                                            li.append((1.0, newstate, r_st + t_reward, True))
                                        else:
                                            # rew = -2.0
                                            li.append((1.0 - per, newstate, r_st, False))

                                            a = 1
                                            newrow, newcol = inc(newrow, newcol, a)
                                            newstate = to_s(newrow, newcol)

                                            if(newrow == 4):
                                                pass

                                            elif (newrow == 3):
                                                newletter = desc[newrow, newcol]

                                                if (newletter == b'G'):
                                                    t_reward = r_st + r_goal
                                                    li.append((per, newstate, t_reward, True))

                                                elif (newletter == b'H'):
                                                    #t_reward = r_st
                                                    t_reward = r_st + r_drop
                                                    li.append((per, newstate, t_reward, True))
                                                else:
                                                    # rew = -2.0
                                                    li.append((per, newstate, r_st, False))

                                            else:
                                                li.append((per, newstate, r_st, False))


                                    elif (self.map_name == "cliff2"):
                                        if (newletter == b'G'):
                                            li.append((1.0, newstate, r_st + 10, True))
                                        else:
                                            # rew = -2.0
                                            li.append((1.0, newstate, r_st, False))


                                    elif (self.map_name == "DST_10*11"):

                                        t_reward = 0

                                        if (newletter == b'G'):
                                            if (newcol == 0):
                                                rew = 1.0
                                                # pass
                                            elif (newcol == 1):
                                                rew = 2.0
                                            elif (newcol == 2):
                                                rew = 3.0
                                            elif (newcol == 4):
                                                rew = 5.0
                                            elif (newcol == 4):
                                                rew = 8.0
                                            elif (newcol == 5):
                                                rew = 16
                                            elif (newcol == 6):
                                                rew = 24
                                            elif (newcol == 7):
                                                rew = 50
                                            elif (newcol == 8):
                                                rew = 74
                                            elif (newcol == 9):
                                                rew = 124
                                            #rew = 1.0

                                            t_reward = w1 * r_st + (1 - w1) * rew
                                            li.append((1.0, newstate, t_reward, True))

                                        else:
                                            # rew = -2.0
                                            li.append((1.0, newstate, r_st * w1, False))

                                    else:
                                        if (newletter == b'G'):
                                            li.append((1.0, newstate, 0, True))
                                        else:
                                            # rew = -2.0
                                            li.append((1.0, newstate, r_st, False))


                                    #elif (self.map_name == "cliff"):

                                        #r = 0.0
                                    #rew = float(newletter == b'G')

                                    # reward = w1 * r_step + (1 - w1) * r

                                    # li.append((1.0, newstate, 0.2, done))

        #show transitions
        if (True):
            for i in range(nS):
                print(i, ":")
                for j in range(4):
                    print(P[i][j])
                    if (j == 3):
                        print("\n")

        super(DeepSeaEnv, self).__init__(nS, nA, P, isd)

    # def step(self, a):
    #
    #     if (self.map_name == "cliff"):
    #         x, r, done, info = super(DeepSeaEnv, self).step(a)
    #
    #         ran = np.random.random()
    #         if (ran < 0.3):
    #             if (x < 36):
    #                 # if (x != 36 and self.get_letter(x) != b'H' and self.get_letter(x) != b'G'):
    #                 return x + 12, r, done, info
    #             else:
    #                 return x, r, done, info
    #         else:
    #             return x, r, done, info
    #
    #     else:
    #         return super(DeepSeaEnv, self).step(a)


    def get_letter(self, n):
        if (n >= self.nS):
            return

        row = n // self.ncol
        col = n % self.ncol
        #print(n)
        letter = self.desc[row, col]

        return letter

    def get_row_col(self, n):
        if (n >= self.nS):
            return

        row = n // self.ncol
        col = n % self.ncol

        return row, col


    def reset(self):
        a = super(DeepSeaEnv, self).reset()
        #super(DeepSeaEnv, self).reset()
        return a
        #print(self.s)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()