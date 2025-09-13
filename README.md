# POLICY EVALUATION

## AIM
To implement and evaluate a given policy in a reinforcement learning environment using the iterative policy evaluation algorithm, and to analyze the resulting state-value function for different policies.

## PROBLEM STATEMENT
In reinforcement learning, the objective is to estimate the value function of a policy, which represents the expected long-term reward for each state when following that policy. The problem is to implement policy evaluation, apply it to different policies in a given environment (e.g., FrozenLake), compute their state-value functions, and compare the performance of the policies based on their values.

## POLICY EVALUATION FUNCTION
```python
Name: HIRUTHIK SUDHAKAR
Register Number: 212223240054

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P))
    while True:
        V = np.zeros(len(P))
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V
```

## OUTPUT
### First policy
<img width="716" height="363" alt="image" src="https://github.com/user-attachments/assets/80f84a37-3d44-4c36-8761-d6e141d7f8d8" />

<img width="679" height="96" alt="image" src="https://github.com/user-attachments/assets/6ca97927-6b75-42c1-aa4f-ff3d598060c7" />

<img width="591" height="151" alt="image" src="https://github.com/user-attachments/assets/960a2733-97a6-4d25-b9eb-00cc91a85225" />


### Second policy
<img width="742" height="438" alt="image" src="https://github.com/user-attachments/assets/f26563ec-0a18-422d-9eff-7d15885c4a26" />

<img width="684" height="105" alt="image" src="https://github.com/user-attachments/assets/6144a2b0-4330-4e29-9b43-ad4028c61d44" />

<img width="603" height="140" alt="image" src="https://github.com/user-attachments/assets/0bf03a33-8a76-4165-9a69-505f9aeb735f" />


### Comparing first and second
<img width="481" height="137" alt="image" src="https://github.com/user-attachments/assets/81f51824-f441-46c2-b99d-18985be90bac" />


## RESULT
The policy evaluation experiment was successfully implemented. The state-value functions for two different policies were obtained and compared. The results show that the first policy performs better than the second policy, as it provides higher expected returns in most states.
