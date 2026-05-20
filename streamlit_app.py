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
            padding: 2.5rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 1100px;
            text-align: center;
        }

        h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
        .description { color: var(--secondary-color); margin-bottom: 1.5rem; font-size: 0.9rem; }

        .controls { display: flex; gap: 1rem; justify-content: center; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
        .input-group { display: flex; align-items: center; gap: 0.5rem; }
        input[type="number"] { width: 60px; padding: 0.4rem; border: 1px solid var(--border-color); border-radius: 0.3rem; }
        
        button { padding: 0.5rem 1rem; border: none; border-radius: 0.3rem; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .primary-btn { background-color: var(--primary-color); color: white; }
        .secondary-btn { background-color: var(--secondary-color); color: white; }
        .accent-btn { background-color: #8b5cf6; color: white; }

        .status { margin-bottom: 1rem; font-weight: 600; color: var(--primary-color); min-height: 1.2rem; }

        /* Strategy Sections */
        .strategy-section { margin-bottom: 2rem; padding: 1rem; background: #fdfdfd; border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .section-title { font-size: 1.1rem; color: var(--primary-color); margin-bottom: 1rem; font-weight: 700; border-bottom: 1px solid #ddd; display: inline-block; }

        .grid-wrapper, .result-grids { display: flex; justify-content: center; gap: 1.5rem; margin-top: 0.5rem; flex-wrap: wrap; }
        .grid-item { flex: 1; min-width: 250px; max-width: 320px; }
        .grid-item h3 { margin-bottom: 0.5rem; font-size: 0.85rem; color: var(--secondary-color); }

        .grid-container {
            display: grid;
            gap: 2px;
            background-color: var(--border-color);
            border: 2px solid var(--border-color);
            border-radius: 4px;
            width: 100%;
            aspect-ratio: 1/1;
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
            font-size: 0.7rem;
            font-weight: bold;
            position: relative;
        }
        .cell:hover { background-color: var(--cell-hover); }
        .cell.start { background-color: var(--cell-start); color: white; }
        .cell.end { background-color: var(--cell-end); color: white; }
        .cell.obstacle { background-color: var(--cell-obstacle); color: white; }
        .cell .arrow { font-size: 1.1rem; }
        .cell .value { font-size: 0.6rem; margin-top: 1px; color: #475569; }
        .cell.start .value, .cell.end .value, .cell.obstacle .value { color: white; }

        .result-controls { margin-top: 1rem; display: flex; justify-content: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>GridWorld RL 策略對照</h1>
        <p class="description">隨機策略 (Random) vs. 最佳策略 (Value Iteration)</p>
        
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

        <div id="result-view" style="display: none;">
            <div class="strategy-section">
                <div class="section-title">隨機策略 (Random Strategy)</div>
                <div class="result-grids">
                    <div class="grid-item">
                        <h3>隨機策略矩陣 (Policy)</h3>
                        <div id="random-policy-grid" class="grid-container"></div>
                    </div>
                    <div class="grid-item">
                        <h3>隨機價值矩陣 (Value)</h3>
                        <div id="random-value-grid" class="grid-container"></div>
                    </div>
                </div>
            </div>

            <div class="strategy-section">
                <div class="section-title">最佳策略 (Optimal Strategy)</div>
                <div class="result-grids">
                    <div class="grid-item">
                        <h3>最佳策略矩陣 (Policy)</h3>
                        <div id="optimal-policy-grid" class="grid-container"></div>
                    </div>
                    <div class="grid-item">
                        <h3>最佳價值矩陣 (Value)</h3>
                        <div id="optimal-value-grid" class="grid-container"></div>
                    </div>
                </div>
            </div>

            <div class="result-controls">
                <button id="regen-policy-btn" class="accent-btn">重新生成對照</button>
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
            const randomPolicyGrid = document.getElementById('random-policy-grid');
            const randomValueGrid = document.getElementById('random-value-grid');
            const optimalPolicyGrid = document.getElementById('optimal-policy-grid');
            const optimalValueGrid = document.getElementById('optimal-value-grid');
            const statusMessage = document.getElementById('status-message');
            const selectionView = document.getElementById('selection-view');
            const resultView = document.getElementById('result-view');

            let currentN = 5;
            let startCell = null;
            let endCell = null;
            let obstacles = [];
            let state = 'WAITING';

            const arrowMap = { 'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→' };

            function updateStatus() {
                switch (state) {
                    case 'WAITING': statusMessage.textContent = '請輸入 n 並點擊「產生網格」'; break;
                    case 'SET_START': statusMessage.textContent = '選擇「起始單元格」(S) '; break;
                    case 'SET_END': statusMessage.textContent = '選擇「結束單元格」(E) '; break;
                    case 'SET_OBSTACLES': 
                        const rem = (currentN - 2) - obstacles.length;
                        statusMessage.textContent = `選擇障礙物 (X)，剩 ${rem} 個`; 
                        break;
                    case 'FINISHED': statusMessage.textContent = '完成！計算中...'; break;
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
                    if (currentN - 2 > 0) state = 'SET_OBSTACLES'; else finish();
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
                [randomPolicyGrid, randomValueGrid, optimalPolicyGrid, optimalValueGrid].forEach(g => createGrid(g, currentN));
                sync();
                await runRL();
            }

            function sync() {
                [randomPolicyGrid, randomValueGrid, optimalPolicyGrid, optimalValueGrid].forEach(container => {
                    const cells = container.querySelectorAll('.cell');
                    cells.forEach((cell, i) => {
                        if (i === startCell) { cell.classList.add('start'); cell.querySelector('.label').textContent = 'S'; }
                        else if (i === endCell) { cell.classList.add('end'); cell.querySelector('.label').textContent = 'E'; }
                        else if (obstacles.includes(i)) { cell.classList.add('obstacle'); cell.querySelector('.label').textContent = 'X'; }
                    });
                });
            }

            async function runRL() {
                const n = currentN;
                const gamma = 0.9;
                const reward = -1;
                const threshold = 1e-4;
                const directions = ['UP', 'DOWN', 'LEFT', 'RIGHT'];

                const getNext = (idx, action) => {
                    let r = Math.floor(idx / n), c = idx % n;
                    let nr = r, nc = c;
                    if (action === 'UP') nr--; else if (action === 'DOWN') nr++;
                    else if (action === 'LEFT') nc--; else if (action === 'RIGHT') nc++;
                    if (nr >= 0 && nr < n && nc >= 0 && nc < n) {
                        let nextIdx = nr * n + nc;
                        if (!obstacles.includes(nextIdx)) return nextIdx;
                    }
                    return idx;
                };

                // 1. Random Policy Generation & Evaluation
                let randomPolicy = {};
                const pCells = randomPolicyGrid.querySelectorAll('.cell');
                pCells.forEach((cell, i) => {
                    if (i === endCell || obstacles.includes(i)) return;
                    const dir = directions[Math.floor(Math.random() * 4)];
                    randomPolicy[i] = dir;
                    cell.querySelector('.arrow').textContent = arrowMap[dir];
                });

                let Vr = new Array(n * n).fill(0);
                for (let iter = 0; iter < 1000; iter++) {
                    let delta = 0;
                    let Vr_new = [...Vr];
                    for (let i = 0; i < n * n; i++) {
                        if (i === endCell || obstacles.includes(i)) continue;
                        const nextIdx = getNext(i, randomPolicy[i]);
                        const newVal = reward + gamma * Vr[nextIdx];
                        Vr_new[i] = newVal;
                        delta = Math.max(delta, Math.abs(Vr[i] - newVal));
                    }
                    Vr = Vr_new;
                    if (delta < threshold) break;
                }

                // 2. Value Iteration (Optimal)
                let Vo = new Array(n * n).fill(0);
                for (let iter = 0; iter < 1000; iter++) {
                    let delta = 0;
                    let Vo_new = [...Vo];
                    for (let i = 0; i < n * n; i++) {
                        if (i === endCell || obstacles.includes(i)) continue;
                        let maxVal = -Infinity;
                        directions.forEach(dir => {
                            const nextIdx = getNext(i, dir);
                            maxVal = Math.max(maxVal, reward + gamma * Vo[nextIdx]);
                        });
                        Vo_new[i] = maxVal;
                        delta = Math.max(delta, Math.abs(Vo[i] - maxVal));
                    }
                    Vo = Vo_new;
                    if (delta < threshold) break;
                }

                // Optimal Policy derivation
                let optimalPolicy = {};
                const opCells = optimalPolicyGrid.querySelectorAll('.cell');
                opCells.forEach((cell, i) => {
                    if (i === endCell || obstacles.includes(i)) return;
                    let bestDir = 'UP', bestVal = -Infinity;
                    directions.forEach(dir => {
                        const nextIdx = getNext(i, dir);
                        const val = reward + gamma * Vo[nextIdx];
                        if (val > bestVal) { bestVal = val; bestDir = dir; }
                    });
                    optimalPolicy[i] = bestDir;
                    cell.querySelector('.arrow').textContent = arrowMap[bestDir];
                });

                // Update Value Grids
                const rvCells = randomValueGrid.querySelectorAll('.cell');
                const ovCells = optimalValueGrid.querySelectorAll('.cell');
                rvCells.forEach((cell, i) => {
                    if (i === endCell) cell.querySelector('.value').textContent = '0.00';
                    else if (!obstacles.includes(i)) cell.querySelector('.value').textContent = Vr[i].toFixed(2);
                });
                ovCells.forEach((cell, i) => {
                    if (i === endCell) cell.querySelector('.value').textContent = '0.00';
                    else if (!obstacles.includes(i)) cell.querySelector('.value').textContent = Vo[i].toFixed(2);
                });

                statusMessage.textContent = '對照完成！';
            }

            generateBtn.onclick = () => {
                const n = parseInt(gridSizeInput.value);
                if (n < 5 || n > 9) return alert('5-9');
                currentN = n; startCell = null; endCell = null; obstacles = [];
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

components.html(html_code, height=900, scrolling=True)

st.sidebar.markdown(\"\"\"
# GridWorld RL 策略對照
這是一個視覺化的強化學習網格世界，對照 **隨機策略** 與 **最佳策略 (Value Iteration)**。

1. **設定**: 指定 $n \\times n$ (5-9)。
2. **操作**: 點擊網格依序設定 **S (起點)**, **E (終點)**, **X (障礙物)**。
3. **對照**: 設定完成後，JS 會在瀏覽器端同時計算：
   - **隨機策略**的價值評估。
   - **最佳策略** (透過 Value Iteration) 及其收斂價值。
\"\"\")
