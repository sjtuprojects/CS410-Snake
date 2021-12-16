import math
import time

import numpy as np
import simUtils as util

EPS = 1e-8

#THIS IS THE 6 PLAYER VERSION
class MCTS():
    def __init__(self, state, nnet, timeLimit=1.9, cpuct=1.0, stepLimit = 100, useTemp = False):
        self.state = state
        self.nnet = nnet
        self.cpuct = cpuct
        self.timeLimit = timeLimit
        #print(timeLimit)
        self.stepLimit = stepLimit
        self.useTemp = useTemp  #flag to use temp or not
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, state, temp=1):
        start_time = time.time()
        while time.time() - start_time < self.timeLimit:
            self.search(state)

        s = util.string_representation(state)
        i = state["controlled_snake_index"]
        counts = [
            self.Nsa[(s, i, a)] if (s, i, a) in self.Nsa else 0
            for a in range(util.get_action_size())
        ]
        #print(counts)

        if self.useTemp:
            if temp == 0:
                bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
                bestA = np.random.choice(bestAs)
                probs = [0] * len(counts)
                probs[bestA] = 1
                return probs

            counts = [x ** (1. / temp) for x in counts]

        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state):
        s = util.string_representation(state)

        if not "steps" in state:
            state["steps"] = 0

        if s not in self.Es:
            dead_snakes = util.are_snakes_dead(state)
            self.Es[s] = 0
            if dead_snakes or state["steps"] >= self.stepLimit:
                # terminal node
                self.Es[s] = util.evaluation(state, dead_snakes)
        if self.Es[s] != 0:
            return self.Es[s]
        
        if s not in self.Ns:
            values = util.evaluation(state, []) #placeholder

            for i in range(2,8):
                # leaf node
                #self.Ps[s], v = self.nnet.predict(state, i)
                #placeholder
                self.Ps[s, i] = util.greedy(state,i)
                valids = util.get_legal_actions_single(state, i)
                self.Ps[s, i] = [a*b for a,b in zip(self.Ps[s, i],valids)] # masking invalid moves 
                sum_Ps_s = np.sum(self.Ps[s, i])
                #print(valids, self.Ps[s, i])
                if sum_Ps_s > 0:
                    self.Ps[s, i] /= sum_Ps_s  # renormalize
                else:
                    #no valid moves
                    self.Ps[s, i] = [1 / len(self.Ps[s, i])]*util.get_action_size()

                self.Vs[s, i] = valids
                self.Ns[s] = 0
            return values

        best_acts = [None] * 6
        for i in range(2,8):
            valids = self.Vs[s, i]
            cur_best = -float('inf')
            best_act = 0

            # pick the action with the highest upper confidence bound
            for a in range(util.get_action_size()):
                if valids[a]:
                    if (s, i, a) in self.Qsa:
                        u = self.Qsa[(s, i, a)] + self.cpuct * self.Ps[s, i][a] * math.sqrt(self.Ns[s]) / (
                                1 + self.Nsa[(s, i, a)])
                    else:
                        u = self.cpuct * self.Ps[s,i][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = a
            
            best_acts[i-2] = best_act

        next_s = util.get_successor(state, best_acts)

        values = self.search(next_s)

        for i in range(2,8):
            a = best_acts[i-2]
            if i in [2,3,4]: v = values[0]
            else: v = values[1]

            if (s, i, a) in self.Qsa:
                self.Qsa[(s, i, a)] = (self.Nsa[(s, i, a)] * self.Qsa[
                    (s, i, a)] + v) / (self.Nsa[(s, i, a)] + 1)
                self.Nsa[(s, i, a)] += 1

            else:
                self.Qsa[(s, i, a)] = v
                self.Nsa[(s, i, a)] = 1
                #print(s,i,a)

        self.Ns[s] += 1
        return values
