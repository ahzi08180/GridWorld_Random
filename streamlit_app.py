import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="GridWorld RL Policy Evaluation")

# The entire frontend and logic in one HTML string
html_code = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GridWorld 網格建造器 - Streamlit 版</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --secondary-color: #64748b;
            --secondary-hover: #475569;
            --bg-color: #f8fafc;
            --container-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
            --cell-empty: #ffffff;
            --cell-start: #22c55e;
            --cell-end: #ef4444;
            --cell-obstacle: #94a3b8;
            --cell-hover: #f1f5f9;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            display: flex;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background-color: var(--container-bg);
            padding: 0.5rem 1rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 800px;
            text-align: center;
        }

        h1 { font-size: 1.3rem; margin-bottom: 0.1rem; }
        .description { color: var(--secondary-color); margin-bottom: 0.5rem; font-size: 0.8rem; }

        .controls { display: flex; gap: 1rem; justify-content: center; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
        .input-group { display: flex; align-items: center; gap: 0.5rem; }
        input[type="number"] { width: 60px; padding: 0.4rem; border: 1px solid var(--border-color); border-radius: 0.3rem; }
        
        button { padding: 0.5rem 1rem; border: none; border-radius: 0.3rem; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .primary-btn { background-color: var(--primary-color); color: white; }
        .secondary-btn { background-color: var(--secondary-color); color: white; }
        .accent-btn { background-color: #8b5cf6; color: white; }
        .accent-btn:disabled { background-color: #cbd5e1; cursor: not-allowed; }

        .status { margin-bottom: 1rem; font-weight: 600; color: var(--primary-color); min-height: 1.2rem; }

        .grid-wrapper, .result-grids { display: flex; justify-content: center; gap: 1rem; margin-top: 0.2rem; flex-wrap: wrap; }
        .grid-item { flex: 1; min-width: 250px; max-width: 300px; }
        .grid-item h3 { margin-bottom: 0.3rem; font-size: 0.9rem; color: var(--secondary-color); }

        .grid-container {
            display: grid;
            gap: 2px;
            background-color: var(--border-color);
            border: 2px solid var(--border-color);
            border-radius: 4px;
            width: 100%;
            aspect-ratio: 1/1;
            max-width: 300px;
            margin: 0 auto;
        }

        .cell {
            background-color: var(--cell-empty);
            aspect-ratio: 1/1;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-size: 0.8rem;
            font-weight: bold;
            position: relative;
        }
        .cell:hover { background-color: var(--cell-hover); }
        .cell.start { background-color: var(--cell-start); color: white; }
        .cell.end { background-color: var(--cell-end); color: white; }
        .cell.obstacle { background-color: var(--cell-obstacle); color: white; }
        .cell .arrow { font-size: 1.2rem; }
        .cell .value { font-size: 0.7rem; margin-top: 2px; color: #475569; }
        .cell.start .value, .cell.end .value, .cell.obstacle .value { color: white; }

        .result-controls { margin-top: 1.5rem; display: flex; justify-content: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>GridWorld 網格建造器 (Streamlit)</h1>
        <p class="description">設定維度 n，指定起點 (S)、終點 (E) 及障礙物 (X)。</p>
        
        <div class="controls">
            <div class="input-group">
                <label>n (5-9):</label>
                <input type="number" id="grid-size" min="5" max="9" value="5">
            </div>
            <button id="generate-btn" class="primary-btn">產生網格</button>
            <button id="reset-btn" class="secondary-btn">重置</button>
        </div>

        <div id="status-message" class="status">請輸入 n 並點擊「產生網格」</div>

        <div id="selection-view" class="grid-wrapper">
            <div id="grid-container" class="grid-container"></div>
        </div>

        <div id="result-view" class="result-view" style="display: none;">
            <div class="result-grids">
                <div class="grid-item">
                    <h3>最佳政策 (Optimal Policy)</h3>
                    <div id="policy-grid" class="grid-container"></div>
                </div>
                <div class="grid-item">
                    <h3>價值函數 (V(s))</h3>
                    <div id="value-grid" class="grid-container"></div>
                </div>
            </div>
            <div class="result-controls">
                <button id="regen-policy-btn" class="accent-btn">重新執行價值迭代</button>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const gridSizeInput = document.getElementById('grid-size');
            const generateBtn = document.getElementById('generate-btn');
            const regenPolicyBtn = document.getElementById('regen-policy-btn');
            const resetBtn = document.getElementById('reset-btn');
            const gridContainer = document.getElementById('grid-container');
            const policyGrid = document.getElementById('policy-grid');
            const valueGrid = document.getElementById('value-grid');
            const statusMessage = document.getElementById('status-message');
            const selectionView = document.getElementById('selection-view');
            const resultView = document.getElementById('result-view');

            let currentN = 5;
            let startCell = null;
            let endCell = null;
            let obstacles = [];
            let policy = {};
            let state = 'WAITING';

            const arrowMap = { 'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→' };

            function updateStatus() {
                switch (state) {
                    case 'WAITING': statusMessage.textContent = '請輸入 n 並點擊「產生網格」'; break;
                    case 'SET_START': statusMessage.textContent = '選擇「起始單元格」(S)'; break;
                    case 'SET_END': statusMessage.textContent = '選擇「結束單元格」(E)'; break;
                    case 'SET_OBSTACLES': 
                        const rem = (currentN - 2) - obstacles.length;
                        statusMessage.textContent = `選擇障礙物 (X)，剩 ${rem} 個`; 
                        break;
                    case 'FINISHED': statusMessage.textContent = '完成！自動執行價值迭代中...'; break;
                }
            }

            function createGrid(container, n, isInteractive = false) {
                container.innerHTML = '';
                container.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
                container.style.gridTemplateRows = `repeat(${n}, 1fr)`;
                for (let i = 0; i < n * n; i++) {
                    const cell = document.createElement('div');
                    cell.classList.add('cell');
                    cell.innerHTML = '<span class="label"></span><span class="arrow"></span><span class="value"></span>';
                    if (isInteractive) cell.onclick = () => handleCellClick(i, cell);
                    container.appendChild(cell);
                }
            }

            function handleCellClick(index, cell) {
                const label = cell.querySelector('.label');
                if (state === 'SET_START') {
                    startCell = index; cell.classList.add('start'); label.textContent = 'S';
                    state = 'SET_END';
                } else if (state === 'SET_END') {
                    if (index === startCell) return;
                    endCell = index; cell.classList.add('end'); label.textContent = 'E';
                    if (currentN > 2) state = 'SET_OBSTACLES'; else finish();
                } else if (state === 'SET_OBSTACLES') {
                    if (index === startCell || index === endCell || obstacles.includes(index)) return;
                    obstacles.push(index); cell.classList.add('obstacle'); label.textContent = 'X';
                    if (obstacles.length >= currentN - 2) finish();
                }
                updateStatus();
            }

            async function finish() {
                state = 'FINISHED';
                selectionView.style.display = 'none';
                resultView.style.display = 'block';
                createGrid(policyGrid, currentN);
                createGrid(valueGrid, currentN);
                sync();
                await runRL();
            }

            function sync() {
                [policyGrid, valueGrid].forEach(container => {
                    const cells = container.querySelectorAll('.cell');
                    cells.forEach((cell, i) => {
                        if (i === startCell) { cell.classList.add('start'); cell.querySelector('.label').textContent = 'S'; }
                        else if (i === endCell) { cell.classList.add('end'); cell.querySelector('.label').textContent = 'E'; }
                        else if (obstacles.includes(i)) { cell.classList.add('obstacle'); cell.querySelector('.label').textContent = 'X'; }
                    });
                });
            }

            async function runRL() {
                const directions = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
                const n = currentN;
                const gamma = 0.9;
                const stepReward = -1;
                const goalReward = 10;
                const threshold = 1e-4;
                let V = new Array(n * n).fill(0);
                let policy = {};

                const getNext = (idx, action) => {
                    let r = Math.floor(idx / n), c = idx % n;
                    let nr = r, nc = c;
                    if (action === 'UP') nr--; else if (action === 'DOWN') nr++;
                    else if (action === 'LEFT') nc--; else if (action === 'RIGHT') nc++;
                    if (nr >= 0 && nr < n && nc >= 0 && nc < n && !obstacles.includes(nr * n + nc)) return nr * n + nc;
                    return idx;
                };

                // Value Iteration
                for (let iter = 0; iter < 1000; iter++) {
                    let delta = 0;
                    let V_new = [...V];
                    for (let i = 0; i < n * n; i++) {
                        if (i === endCell || obstacles.includes(i)) continue;
                        let maxVal = -Infinity;
                        let bestAction = null;
                        for (let action of directions) {
                            const nextIdx = getNext(i, action);
                            const reward = (nextIdx === endCell) ? goalReward : stepReward;
                            const val = reward + gamma * V[nextIdx];
                            if (val > maxVal) {
                                maxVal = val;
                                bestAction = action;
                            }
                        }
                        V_new[i] = maxVal;
                        policy[i] = bestAction;
                        delta = Math.max(delta, Math.abs(V[i] - maxVal));
                    }
                    V = V_new;
                    if (delta < threshold) break;
                }

                // Display optimal policy
                const pCells = policyGrid.querySelectorAll('.cell');
                pCells.forEach((cell, i) => {
                    if (i === endCell || obstacles.includes(i)) return;
                    const action = policy[i];
                    cell.querySelector('.arrow').textContent = arrowMap[action];
                });

                // Display value function
                const vCells = valueGrid.querySelectorAll('.cell');
                vCells.forEach((cell, i) => {
                    if (i === endCell) cell.querySelector('.value').textContent = '0.00';
                    else if (!obstacles.includes(i)) cell.querySelector('.value').textContent = V[i].toFixed(2);
                });
                statusMessage.textContent = '價值迭代完成！最佳政策與 V(s) 已更新。';
            }

            generateBtn.onclick = () => {
                const n = parseInt(gridSizeInput.value);
                if (n < 5 || n > 9) return alert('5-9');
                currentN = n; startCell = null; endCell = null; obstacles = []; policy = {};
                state = 'SET_START';
                selectionView.style.display = 'flex';
                resultView.style.display = 'none';
                createGrid(gridContainer, n, true);
                updateStatus();
            };
            regenPolicyBtn.onclick = runRL;
            resetBtn.onclick = () => generateBtn.click();
        });
    </script>
</body>
</html>
"""

components.html(html_code, height=700, scrolling=True)

st.sidebar.markdown("""
# GridWorld RL
這是一個視覺化的強化學習網格世界，使用價值迭代算法推導最佳政策。
1. **設定**: 指定 $n \\times n$。
2. **操作**: 點擊網格依序設定 S (起點), E (終點), X (障礙物)。
3. **自動計算**: 設定完成後，JS 會在瀏覽器端直接計算價值迭代，顯示最佳政策與價值函數。
""")
