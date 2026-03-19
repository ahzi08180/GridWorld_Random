from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    n = data['n']
    start = data['start']
    end = data['end']
    obstacles = data['obstacles']

    # Parameters
    gamma = 0.9
    threshold = 1e-4
    step_reward = -1
    goal_reward = 10
    
    # Initialize V
    V = np.zeros(n * n)
    policy = {}
    
    # Directions: 0: Up, 1: Down, 2: Left, 3: Right
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def get_next_state(s, action_name):
        r, c = divmod(s, n)
        dr, dc = 0, 0
        if action_name == 'UP': dr = -1
        elif action_name == 'DOWN': dr = 1
        elif action_name == 'LEFT': dc = -1
        elif action_name == 'RIGHT': dc = 1
        nr, nc = r + dr, c + dc
        
        # Check boundaries
        if 0 <= nr < n and 0 <= nc < n:
            next_s = nr * n + nc
            # Check obstacles
            if next_s in obstacles:
                return s
            return next_s
        return s

    # Value Iteration
    while True:
        delta = 0
        new_V = np.copy(V)
        new_policy = {}
        for s in range(n * n):
            if s == end or s in obstacles:
                continue
            
            max_val = -np.inf
            best_action = None
            for action in actions:
                next_s = get_next_state(s, action)
                reward = goal_reward if next_s == end else step_reward
                val = reward + gamma * V[next_s]
                if val > max_val:
                    max_val = val
                    best_action = action
            
            new_V[s] = max_val
            new_policy[str(s)] = best_action
            delta = max(delta, abs(V[s] - max_val))
        
        V = new_V
        policy = new_policy
        if delta < threshold:
            break

    return jsonify({'values': V.tolist(), 'policy': policy})

if __name__ == '__main__':
    app.run(debug=True)
