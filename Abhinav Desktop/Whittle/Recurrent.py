import numpy as np

def is_recurrent(P, s):
    n = P.shape[0]
    visited = [False] * n
    
    def dfs(state):
        if visited[state]:
            return False
        visited[state] = True
        for next_state in range(n):
            if P[state, next_state] > 0:
                if next_state == s or dfs(next_state):
                    return True
        return False
    
    return dfs(s)

# Example usage:
P = np.array([[0.1, 0.9, 0, 0, 0],
        [0.1, 0, 0.9, 0, 0],
        [0.1, 0, 0, 0.9, 0],
        [0.1, 0, 0, 0, 0.9],
        [0.1, 0, 0, 0, 0.9]])
# s = 4
# print(is_recurrent(P, s))


Q = np.array([[1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]])

# s = 0
# print(is_recurrent(Q, s))

R = np.array([[1, 0, 0, 0, 0],
        [0.1, 0, 0.9, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0.1, 0, 0, 0, 0.9]])

# s = 4
# print(is_recurrent(R, s))

T = np.array([[0.5,0,0,0.5], [0.5,0.5,0,0], [0,0.5,0.5,0], [0,0,0.5,0.5]])
s = 3
print(is_recurrent(T, s))