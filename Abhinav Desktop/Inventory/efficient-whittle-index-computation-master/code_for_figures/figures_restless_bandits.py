import numpy as np
import itertools
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='notebook', style='darkgrid', palette='bright', font='sans-serif', font_scale=2, color_codes=True)

def dense_probability_matrix(n):
    P = np.random.exponential(1,[n,n])
    for line in P:
        line /= sum(line)
    return P

class MarkovRewardProcess:
    def __init__(self, P, R):
        self.P = P
        self.R = R

    def average_reward(self):
        n = len(self.P)
        A = np.eye(n)
        A[1:,0] += 1
        A[:,1:] -= self.P[:,1:]
        return np.linalg.inv(A)@self.R
        
class RestlessBandit:
    def __init__(self, P0, P1, R0, R1):
        self.P0 = P0
        self.P1 = P1
        self.R0 = R0
        self.R1 = R1
    
    @staticmethod
    def random_restless_bandit(n, seed=None):
        np.random.seed(seed)
        P0 = dense_probability_matrix(n)
        P1 = dense_probability_matrix(n)
        R0 = np.random.rand(n)
        R1 = np.random.rand(n)
        return RestlessBandit(P0, P1, R0, R1)

    def transition_matrix(self, activation_vector):
        n = len(self.P0)
        P = np.zeros(shape=(n,n))
        for i in range(n):
            P[i] = self.P0[i]*(1-activation_vector[i]) + self.P1[i]*activation_vector[i]
        return P

    def rewards(self, activation_vector):
        n = len(self.P0)
        R = np.zeros(shape=(n))
        for i in range(n):
            R[i] = self.R0[i]*(1-activation_vector[i]) + self.R1[i]*activation_vector[i]
        return R

    def average_reward(self, activation_vector, tax, output_biais=False):
        P = self.transition_matrix(activation_vector)
        R = self.rewards(activation_vector) - tax*np.array(activation_vector)
        if not output_biais:
            """Output the gain"""
            return MarkovRewardProcess(P, R).average_reward()[0]
        else:
            """Output the bias vector (gain at the 1st component)"""
            return MarkovRewardProcess(P, R).average_reward()
    
    def compute_reward_all_policies(self, xlim=[0, 1], output_biais=False):
        n = len(self.P0)
        names_of_policies = []
        rewards_of_policies = []
        for size in range(n+1):
            for rest_set in itertools.combinations(range(n), size):
                activation_vector = np.ones(n)
                for i in rest_set:
                    activation_vector[i] = 0
                r_left = self.average_reward(activation_vector, tax=xlim[0], output_biais=output_biais)
                r_right = self.average_reward(activation_vector, tax=xlim[1], output_biais=output_biais)
                rewards_of_policies.append([r_left, r_right])
                names_of_policies.append(rest_set)
        return names_of_policies, rewards_of_policies, xlim

    def figure1(self, xlim=[0, 1], yplace_of_labels=1):
        names_of_policies, rewards_of_policies, _ = self.compute_reward_all_policies(xlim)
        for name,reward in zip(names_of_policies, rewards_of_policies):
            plt.plot(xlim, reward, 'k:')
            for i in [0,1]:
                if len(name) == 0:
                    plt.text(xlim[i], reward[i], '{ }', clip_on=True)
                else:
                    rest_set = np.array(name) + 1
                    plt.text(xlim[i], reward[i], set(rest_set), fontsize='large', clip_on=True)
        x_range = np.linspace(xlim[0], xlim[1], 1000)
        y_range = [np.interp(x_range, xlim, rewards) for rewards in rewards_of_policies]
        best_rest_sets = [names_of_policies[i] for i in np.argmax(y_range, axis=0)]
        best_y_range = np.max(y_range, axis=0)
        plt.plot(x_range, best_y_range, 'r', label='Optimal Gain', lw=3)
        plt.xlabel('Tax')
        plt.ylabel('Average Gain')
        
        indices, policies, states = self.index_from_best_rest_sets(best_rest_sets)
        for i in range(len(indices)-2):
            if i<len(indices)-2:
                plt.axvline(x_range[indices[i+1]], color='r', ls='--', lw=2)
                plt.text(x_range[indices[i+1]], yplace_of_labels*0.635, "$\lambda_{}$".format(states[i]),color='r', ha='center', fontsize='xx-large')
                plt.text(x_range[indices[i+1]]-0.01, yplace_of_labels*0.98, "$\sigma_{}={}$".format(i+1, states[i]),color='r', ha='center', fontsize='xx-large', rotation="vertical")
            if len(policies[i]) == 0:
                rest_set = "$\pi^1=\{ \}$"
            else:
                rest_set = "$\pi^{}=$".format(i+1)+str(set(np.array(policies[i])+1))
            plt.text((x_range[indices[i]]+x_range[indices[i+1]])/2, yplace_of_labels*0.92, rest_set, ha='center', color='r', fontsize='xx-large', rotation="vertical")
        xx = []
        yy = []
        for i in range(0, len(indices)-1):
            print(policies[i])
            xx = np.append(xx, x_range[indices[i]:indices[i+1]])
            yy = np.append(yy, y_range[names_of_policies.index(policies[i])][indices[i]:indices[i+1]])
        #plt.plot(xx, yy, color='b', ls=':', label='Algorithm', lw=3.5)
        plt.legend(loc='best')
        return [x_range[indices[i+1]] for i in range(len(states))], states
    
    def figure3(self, xlim=[0, 1]):
        names_of_policies, rewards_of_policies, _ = self.compute_reward_all_policies(xlim)
        for name,reward in zip(names_of_policies, rewards_of_policies):
            plt.plot(xlim, reward, 'k:')
            for i in [0,1]:
                if len(name) == 0:
                    plt.text(xlim[i], reward[i], '{ }', clip_on=True)
                else:
                    rest_set = np.array(name) + 1
                    plt.text(xlim[i], reward[i], set(rest_set), fontsize='large', clip_on=True)
        x_range = np.linspace(xlim[0], xlim[1], 1000)
        y_range = [np.interp(x_range, xlim, rewards) for rewards in rewards_of_policies]
        best_rest_sets = [names_of_policies[i] for i in np.argmax(y_range, axis=0)]
        best_y_range = np.max(y_range, axis=0)
        plt.plot(x_range, best_y_range, 'r', label='Optimal Gain', lw=3)
        plt.xlabel('Tax')
        plt.ylabel('Average gain')
        
        indices, policies, states = self.index_from_best_rest_sets(best_rest_sets)
        for i in range(len(indices)-2):
            if i<len(indices)-2:
                plt.axvline(x_range[indices[i+1]], color='b', ls='--', lw=2)
                plt.text(x_range[indices[i+1]], -0.015, "$\lambda_{}$".format(states[i]),color='b', ha='center', fontsize='xx-large')
                plt.text(x_range[indices[i+1]]-0.01, 0.185, "$\sigma_{}={}$".format(i+1, states[i]),color='b', ha='center', fontsize='xx-large', rotation="vertical")
            if len(policies[i]) == 0:
                rest_set = "$\pi^1=\{ \}$"
            else:
                rest_set = "$\pi^{}=$".format(i+1)+str(set(np.array(policies[i])+1))
            plt.text((x_range[indices[i]]+x_range[indices[i+1]])/2, 0.06, rest_set, ha='center', color='b', fontsize='xx-large', rotation="vertical")
        xx = []
        yy = []
        for i in range(0, len(indices)-1):
            print(policies[i])
            xx = np.append(xx, x_range[indices[i]:indices[i+1]])
            yy = np.append(yy, y_range[names_of_policies.index(policies[i])][indices[i]:indices[i+1]])
        plt.plot(xx, yy, color='b', ls='-.', label='Algorithm', lw=3.5)
        plt.legend(loc='best')
        return [x_range[indices[i+1]] for i in range(len(states))], states
    
    def figure3bis(self, xlim=[0, 1]):
        names_of_policies, rewards_of_policies, _ = self.compute_reward_all_policies(xlim)
        for name,reward in zip(names_of_policies, rewards_of_policies):
            plt.plot(xlim, reward, 'k:')
            for i in [0,1]:
                if len(name) == 0:
                    plt.text(xlim[i], reward[i], '{ }', clip_on=True)
                else:
                    rest_set = np.array(name) + 1
                    plt.text(xlim[i], reward[i], set(rest_set), fontsize='large', clip_on=True)
        x_range = np.linspace(xlim[0], xlim[1], 1000)
        y_range = [np.interp(x_range, xlim, rewards) for rewards in rewards_of_policies]
        best_rest_sets = [names_of_policies[i] for i in np.argmax(y_range, axis=0)]
        best_y_range = np.max(y_range, axis=0)
        plt.plot(x_range, best_y_range, 'r', label='Optimal', lw=3)
        plt.xlabel('Tax')
        plt.ylabel('Average reward')
        
        indices, policies, states = self.index_from_best_rest_sets(best_rest_sets)
        for i in range(len(indices)-2):
            if i<len(indices)-2:
                plt.axvline(x_range[indices[i+1]], color='b', ls='-.', lw=2)
                plt.text(x_range[indices[i+1]], -0.011, "$\lambda_{}$".format(states[i]),color='b', ha='center', fontsize='xx-large')
                plt.text(x_range[indices[i+1]]-0.001, 0.01, "$\sigma_{}={}$".format(3, states[i]),color='b', ha='center', fontsize='xx-large', rotation="vertical")
            if len(policies[i]) == 0:
                rest_set = "$\pi^1=\{ \}$"
            else:
                rest_set = "$\pi^{}=$".format(3)+str(set(np.array(policies[i])+1))
            plt.text((x_range[indices[i]]+x_range[indices[i+1]])/2, -0.005, rest_set, ha='center', color='b', fontsize='xx-large', rotation="vertical")
        xx = []
        yy = []
        for i in range(0, len(indices)-1):
            print(policies[i])
            xx = np.append(xx, x_range[indices[i]:indices[i+1]])
            yy = np.append(yy, y_range[names_of_policies.index(policies[i])][indices[i]:indices[i+1]])
        plt.plot(xx, yy, color='b', ls='--', label='Algorithm', lw=3.5)
        plt.legend(loc='best')
        return [x_range[indices[i+1]] for i in range(len(states))], states
    
    def figure2a(self, xlim=[0, 1]):
        names_of_policies, rewards_of_policies, _ = self.compute_reward_all_policies(xlim)
        x_range = np.linspace(xlim[0], xlim[1], 1000)
        y_range = [np.interp(x_range, xlim, rewards) for rewards in rewards_of_policies]
        best_rest_sets = [names_of_policies[i] for i in np.argmax(y_range, axis=0)]
        best_y_range = np.max(y_range, axis=0)
        
        indices, policies, states = self.index_from_best_rest_sets(best_rest_sets)
        #for i in range(len(indices)-2):
        for i in range(1):
            if i<len(indices)-2:
                plt.axvline(x_range[indices[i+1]], color='b', ls='--', lw=2)
                plt.text(x_range[indices[i+1]], 0.64, "$\lambda_{}$".format(states[i]),color='b', ha='center', fontsize='xx-large')
                plt.text(x_range[indices[i+1]]-0.01, 0.98, "$\sigma_{}={}$".format(i+1, states[i]),color='b', ha='center', fontsize='xx-large', rotation="vertical")
            if len(policies[i]) == 0:
                rest_set = "$\pi^1=\{ \}$"
            else:
                rest_set = "$\pi^{}=$".format(i+1)+str(set(np.array(policies[i])+1))
            plt.text((x_range[indices[i]]+x_range[indices[i+1]])/2, 0.92, rest_set, ha='center', color='b', fontsize='xx-large', rotation="vertical")
        k = 0
        for name,reward in zip(names_of_policies, rewards_of_policies):
            if len(name) == 0 or len(name) == 1:# or list(name) == [0,1] or list(name) == [0,1,3] or len(name) == 4:
                plt.plot(x_range, y_range[k], 'k:')
                for i in [0]:
                    if len(name) == 0:
                        plt.text(xlim[i], reward[i], '{ }', clip_on=True)
                    else:
                        rest_set = np.array(name) + 1
                        plt.text(xlim[i], reward[i], set(rest_set), fontsize='large', clip_on=True)               
            #else:
            #    plt.plot(x_range[0:indices[1]], y_range[k][0:indices[1]], 'k:')
            k += 1

        plt.plot(x_range[0:indices[1]], best_y_range[0:indices[1]], 'b-.', label='Algorithm Gain', lw=3)
        plt.xlabel('Tax')
        plt.ylabel('Average Gain')
        xx = []
        yy = []
        for i in range(0, len(indices)-1):
            print(policies[i])
            xx = np.append(xx, x_range[indices[i]:indices[i+1]])
            yy = np.append(yy, y_range[names_of_policies.index(policies[i])][indices[i]:indices[i+1]])
        #plt.plot(xx, yy, color='b', ls=':', label='Algorithm', lw=3.5)
        plt.legend(loc='best')
        return [x_range[indices[i+1]] for i in range(len(states))], states
    
    def figure2b(self, xlim=[0, 1]):
        names_of_policies, rewards_of_policies, _ = self.compute_reward_all_policies(xlim)
        x_range = np.linspace(xlim[0], xlim[1], 1000)
        y_range = [np.interp(x_range, xlim, rewards) for rewards in rewards_of_policies]
        best_rest_sets = [names_of_policies[i] for i in np.argmax(y_range, axis=0)]
        best_y_range = np.max(y_range, axis=0)
        
        indices, policies, states = self.index_from_best_rest_sets(best_rest_sets)
        #for i in range(len(indices)-2):
        for i in range(2):
            if i<len(indices)-2:
                plt.axvline(x_range[indices[i+1]], color='b', ls='--', lw=2)
                plt.text(x_range[indices[i+1]], 0.64, "$\lambda_{}$".format(states[i]),color='b', ha='center', fontsize='xx-large')
                plt.text(x_range[indices[i+1]]-0.01, 0.98, "$\sigma_{}={}$".format(i+1, states[i]),color='b', ha='center', fontsize='xx-large', rotation="vertical")
            if len(policies[i]) == 0:
                rest_set = "$\pi^1=\{ \}$"
            else:
                rest_set = "$\pi^{}=$".format(i+1)+str(set(np.array(policies[i])+1))
            plt.text((x_range[indices[i]]+x_range[indices[i+1]])/2, 0.92, rest_set, ha='center', color='b', fontsize='xx-large', rotation="vertical")
        k = 0
        for name,reward in zip(names_of_policies, rewards_of_policies):
            #if len(name) == 0:# or list(name) == [0,1] or list(name) == [0,1,3] or len(name) == 4:
            #    plt.plot(x_range[0:indices[1]], y_range[k][0:indices[1]], 'k:')
            #else:
            #    plt.plot(x_range[0:indices[2]], y_range[k][0:indices[2]], 'k:')
            if len(name) < 3 and (1 in list(name) or list(name) == [1]):
                plt.plot(x_range, y_range[k], 'k:')
                for i in [0]:
                    if len(name) == 0:
                        plt.text(xlim[i], reward[i], '{ }', clip_on=True)
                    else:
                        rest_set = np.array(name) + 1
                        plt.text(xlim[i], reward[i], set(rest_set), fontsize='large', clip_on=True)
            k += 1

        plt.plot(x_range[0:indices[2]], best_y_range[0:indices[2]], 'b-.', label='Algorithm Gain', lw=3)
        plt.xlabel('Tax')
        plt.ylabel('Average Gain')
        xx = []
        yy = []
        for i in range(0, len(indices)-1):
            print(policies[i])
            xx = np.append(xx, x_range[indices[i]:indices[i+1]])
            yy = np.append(yy, y_range[names_of_policies.index(policies[i])][indices[i]:indices[i+1]])
        #plt.plot(xx, yy, color='b', ls=':', label='Algorithm', lw=3.5)
        plt.legend(loc='best')
        return [x_range[indices[i+1]] for i in range(len(states))], states
    
    def figure2c(self, xlim=[0, 1]):
        names_of_policies, rewards_of_policies, _ = self.compute_reward_all_policies(xlim)
        x_range = np.linspace(xlim[0], xlim[1], 1000)
        y_range = [np.interp(x_range, xlim, rewards) for rewards in rewards_of_policies]
        best_rest_sets = [names_of_policies[i] for i in np.argmax(y_range, axis=0)]
        best_y_range = np.max(y_range, axis=0)
        
        indices, policies, states = self.index_from_best_rest_sets(best_rest_sets)
        #for i in range(len(indices)-2):
        for i in range(3):
            if i<len(indices)-2:
                plt.axvline(x_range[indices[i+1]], color='b', ls='--', lw=2)
                plt.text(x_range[indices[i+1]], 0.635, "$\lambda_{}$".format(states[i]),color='b', ha='center', fontsize='xx-large')
                plt.text(x_range[indices[i+1]]-0.01, 0.98, "$\sigma_{}={}$".format(i+1, states[i]),color='b', ha='center', fontsize='xx-large', rotation="vertical")
            if len(policies[i]) == 0:
                rest_set = "$\pi^1=\{ \}$"
            else:
                rest_set = "$\pi^{}=$".format(i+1)+str(set(np.array(policies[i])+1))
            plt.text((x_range[indices[i]]+x_range[indices[i+1]])/2, 0.92, rest_set, ha='center', color='b', fontsize='xx-large', rotation="vertical")
        k = 0
        for name,reward in zip(names_of_policies, rewards_of_policies):
            #if len(name) == 0:# or list(name) == [0,1] or list(name) == [0,1,3] or len(name) == 4:
            #    plt.plot(x_range[0:indices[1]], y_range[k][0:indices[1]], 'k:')
            #else:
            #    plt.plot(x_range[0:indices[2]], y_range[k][0:indices[2]], 'k:')
            if (0 in list(name) and 1 in list(name)) or list(name) == [1]:
                plt.plot(x_range, y_range[k], 'k:')
                for i in [0]:
                    if len(name) == 0:
                        plt.text(xlim[i], reward[i], '{ }', clip_on=True)
                    else:
                        rest_set = np.array(name) + 1
                        plt.text(xlim[i], reward[i], set(rest_set), fontsize='large', clip_on=True)
            k += 1

        plt.plot(x_range[0:indices[3]], best_y_range[0:indices[3]], 'b-.', label='Algorithm Gain', lw=3)
        plt.xlabel('Tax')
        plt.ylabel('Average Gain')
        xx = []
        yy = []
        for i in range(0, len(indices)-1):
            print(policies[i])
            xx = np.append(xx, x_range[indices[i]:indices[i+1]])
            yy = np.append(yy, y_range[names_of_policies.index(policies[i])][indices[i]:indices[i+1]])
        #plt.plot(xx, yy, color='b', ls=':', label='Algorithm', lw=3.5)
        plt.legend(loc='best')
        return [x_range[indices[i+1]] for i in range(len(states))], states
    
    def index_from_best_rest_sets(self, best_rest_sets):
        best_policies = [best_rest_sets[0]]
        indices = [0]
        indices_states = []
        for index, rest_set in enumerate(best_rest_sets[1:]):
            if not (rest_set == best_policies[-1]):
                if set(best_policies[-1]) > set(rest_set) or len(best_policies[-1]) > len(rest_set): # lastly considered rest_set is not a subset of current rest_set
                    #print('non indexable', np.array(arms)+1, np.array(best_policies[-1])+1)
                    pass
                else:
                    state = diff_policy(rest_set,best_policies[-1])
                    if len(best_policies[-1]) == len(rest_set):
                        if state != '':
                            rest_set = list(best_policies[-1])
                            rest_set.append(state)
                            rest_set = tuple(sorted(rest_set))
                    if state != '':
                        indices_states.append(state+1)
                        best_policies.append(rest_set)
                        indices.append(index)
        best_policies.append(best_rest_sets[-1])
        indices.append(index)
        return indices, best_policies, indices_states
                
def diff_policy(policy1, policy2):
    for i in policy1:
        if i not in policy2:
            return i
    return ''
