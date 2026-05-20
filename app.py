from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    n = data['n']
    start = data['start']
    end = data['end']
    obstacles = data['obstacles']
    random_policy = data['random_policy']

    gamma = 0.9
    threshold = 1e-4
    reward_step = -1
    
    actions = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1)
    }

    def get_next_state(s, action_name):
        r, c = divmod(s, n)
        dr, dc = actions[action_name]
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            next_s = nr * n + nc
            if next_s in obstacles:
                return s
            return next_s
        return s

    # 1. Random Policy Evaluation
    V_random = np.zeros(n * n)
    while True:
        delta = 0
        new_V = np.copy(V_random)
        for s in range(n * n):
            if s == end or s in obstacles:
                continue
            action_name = random_policy.get(str(s))
            if not action_name: continue
            next_s = get_next_state(s, action_name)
            v = reward_step + gamma * V_random[next_s]
            new_V[s] = v
            delta = max(delta, abs(V_random[s] - v))
        V_random = new_V
        if delta < threshold:
            break

    # 2. Value Iteration for Optimal Policy
    V_optimal = np.zeros(n * n)
    while True:
        delta = 0
        new_V = np.copy(V_optimal)
        for s in range(n * n):
            if s == end or s in obstacles:
                continue
            
            # Lookahead: find max over all actions
            v_actions = []
            for action_name in actions:
                next_s = get_next_state(s, action_name)
                v_actions.append(reward_step + gamma * V_optimal[next_s])
            
            best_v = max(v_actions)
            new_V[s] = best_v
            delta = max(delta, abs(V_optimal[s] - best_v))
        V_optimal = new_V
        if delta < threshold:
            break

    # Derive Optimal Policy from V_optimal
    optimal_policy = {}
    for s in range(n * n):
        if s == end or s in obstacles:
            continue
        best_action = 'UP'
        best_val = -float('inf')
        for action_name in actions:
            next_s = get_next_state(s, action_name)
            val = reward_step + gamma * V_optimal[next_s]
            if val > best_val:
                best_val = val
                best_action = action_name
        optimal_policy[s] = best_action

    return jsonify({
        'random': {
            'values': V_random.tolist(),
            'policy': random_policy
        },
        'optimal': {
            'values': V_optimal.tolist(),
            'policy': optimal_policy
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
