document.addEventListener('DOMContentLoaded', () => {
    const gridSizeInput = document.getElementById('grid-size');
    const generateBtn = document.getElementById('generate-btn');
    const policyBtn = document.getElementById('policy-btn');
    const evaluateBtn = document.getElementById('evaluate-btn');
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
    let policy = {}; // index: direction
    let state = 'WAITING'; // WAITING, SET_START, SET_END, SET_OBSTACLES, FINISHED

    const arrowMap = {
        'UP': '↑',
        'DOWN': '↓',
        'LEFT': '←',
        'RIGHT': '→'
    };

    function updateStatus() {
        switch (state) {
            case 'WAITING':
                statusMessage.textContent = '請輸入 n 並點擊「產生網格」';
                break;
            case 'SET_START':
                statusMessage.textContent = '請點擊網格以選擇「起始單元格」(綠色)';
                break;
            case 'SET_END':
                statusMessage.textContent = '請點擊網格以選擇「結束單元格」(紅色)';
                break;
            case 'SET_OBSTACLES':
                const remaining = (currentN - 2) - obstacles.length;
                statusMessage.textContent = `請選擇障礙物 (灰色)，剩餘 ${remaining} 個`;
                break;
            case 'FINISHED':
                statusMessage.textContent = '設定完成！已自動生成隨機策略與評估圖。';
                break;
        }
    }

    function createGrid(container, n, isInteractive = false) {
        container.innerHTML = '';
        container.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
        container.style.gridTemplateRows = `repeat(${n}, 1fr)`;
        
        for (let i = 0; i < n * n; i++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.index = i;
            cell.innerHTML = '<span class="label"></span><span class="arrow"></span><span class="value"></span>';
            if (isInteractive) {
                cell.addEventListener('click', () => handleCellClick(i, cell));
            }
            container.appendChild(cell);
        }
    }

    async function handleCellClick(index, cellElement) {
        const label = cellElement.querySelector('.label');
        if (state === 'SET_START') {
            startCell = index;
            cellElement.classList.add('start');
            label.textContent = 'S';
            state = 'SET_END';
        } else if (state === 'SET_END') {
            if (index === startCell) return;
            endCell = index;
            cellElement.classList.add('end');
            label.textContent = 'E';
            if (currentN - 2 > 0) {
                state = 'SET_OBSTACLES';
            } else {
                await finishAndShowResults();
            }
        } else if (state === 'SET_OBSTACLES') {
            if (index === startCell || index === endCell || obstacles.includes(index)) return;
            
            obstacles.push(index);
            cellElement.classList.add('obstacle');
            label.textContent = 'X';

            if (obstacles.length >= currentN - 2) {
                await finishAndShowResults();
            }
        }
        updateStatus();
    }

    async function finishAndShowResults() {
        state = 'FINISHED';
        selectionView.style.display = 'none';
        resultView.style.display = 'block';
        
        // Create all result grids
        [randomPolicyGrid, randomValueGrid, optimalPolicyGrid, optimalValueGrid].forEach(grid => createGrid(grid, currentN));
        
        // Sync selection state to all result grids
        syncGrids();
        
        // Auto generate and solve
        await generateAndSolve();
    }

    function syncGrids() {
        [randomPolicyGrid, randomValueGrid, optimalPolicyGrid, optimalValueGrid].forEach(container => {
            const cells = container.querySelectorAll('.cell');
            cells.forEach((cell, index) => {
                const label = cell.querySelector('.label');
                if (index === startCell) {
                    cell.classList.add('start');
                    label.textContent = 'S';
                } else if (index === endCell) {
                    cell.classList.add('end');
                    label.textContent = 'E';
                } else if (obstacles.includes(index)) {
                    cell.classList.add('obstacle');
                    label.textContent = 'X';
                }
            });
        });
    }

    async function generateAndSolve() {
        const directions = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
        const random_policy = {};
        
        // Clear and generate random policy
        const randomPolicyCells = randomPolicyGrid.querySelectorAll('.cell');
        randomPolicyCells.forEach((cell, index) => {
            const i = parseInt(index);
            if (i === endCell || obstacles.includes(i)) return;
            const randomDir = directions[Math.floor(Math.random() * directions.length)];
            random_policy[i] = randomDir;
            cell.querySelector('.arrow').textContent = arrowMap[randomDir];
        });

        statusMessage.textContent = '正在計算隨機策略與最佳策略...';
        try {
            const response = await fetch('/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    n: currentN, start: startCell, end: endCell,
                    obstacles: obstacles, random_policy: random_policy
                })
            });
            const data = await response.json();
            
            // 1. Update Random Value Grid
            const randomValueCells = randomValueGrid.querySelectorAll('.cell');
            data.random.values.forEach((val, index) => {
                if (index !== endCell && !obstacles.includes(index)) {
                    randomValueCells[index].querySelector('.value').textContent = val.toFixed(2);
                } else if (index === endCell) {
                    randomValueCells[index].querySelector('.value').textContent = '0.00';
                }
            });

            // 2. Update Optimal Policy Grid
            const optimalPolicyCells = optimalPolicyGrid.querySelectorAll('.cell');
            const optPol = data.optimal.policy;
            optimalPolicyCells.forEach((cell, index) => {
                if (optPol[index]) {
                    cell.querySelector('.arrow').textContent = arrowMap[optPol[index]];
                }
            });

            // 3. Update Optimal Value Grid
            const optimalValueCells = optimalValueGrid.querySelectorAll('.cell');
            data.optimal.values.forEach((val, index) => {
                if (index !== endCell && !obstacles.includes(index)) {
                    optimalValueCells[index].querySelector('.value').textContent = val.toFixed(2);
                } else if (index === endCell) {
                    optimalValueCells[index].querySelector('.value').textContent = '0.00';
                }
            });

            statusMessage.textContent = '計算完成！對照如下：';
        } catch (error) {
            console.error('Error:', error);
            statusMessage.textContent = '計算出錯，請檢查後台';
        }
    }

    generateBtn.addEventListener('click', () => {
        const n = parseInt(gridSizeInput.value);
        if (n < 5 || n > 9) {
            alert('維度 n 必須在 5 到 9 之間');
            return;
        }
        currentN = n;
        startCell = null;
        endCell = null;
        obstacles = [];
        policy = {};
        state = 'SET_START';
        
        selectionView.style.display = 'flex';
        resultView.style.display = 'none';
        
        createGrid(gridContainer, n, true);
        updateStatus();
    });

    regenPolicyBtn.addEventListener('click', () => {
        generateAndSolve();
    });

    resetBtn.addEventListener('click', () => {
        generateBtn.click();
    });

    updateStatus();
});
