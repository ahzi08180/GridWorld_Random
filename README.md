# GridWorld Value Iteration & Optimal Policy

這是一個基於 Flask 的 GridWorld 網格世界視覺化工具，用於展示強化學習中的**價值迭代 (Value Iteration)** 與 **最佳政策推導 (Optimal Policy Derivation)**。

## 功能特點
- **自定義網格**：支援 $n \times n$ 維度（5-9），可自由設定起點、終點與障礙物。
- **價值迭代算法**：使用價值迭代算法自動計算最佳價值函數和最佳政策。
- **雙圖並行展示**：
  - **左圖**：展示推導出的最佳政策（方向箭頭）。
  - **右圖**：展示計算出的狀態價值函數 $V(s)$。
- **互動式體驗**：支援即時重新執行價值迭代，觀察結果變化。

## 專案展示
![GridWorld Demo](https://gridworldrandom-gpywymxsy6njquyjluhf6g.streamlit.app/)

## 技術堆疊
- **Backend**: Python, Flask, NumPy
- **Frontend**: Vanilla JavaScript, CSS3 (Modern HSL palette), HTML5

## 如何在本機執行
### 方法一：使用 Flask (傳統版)
1. **安裝依賴**：`pip install -r requirements.txt`
2. **啟動伺服器**：`python app.py`
3. **瀏覽網頁**：`http://127.0.0.1:5000`

### 方法二：使用 Streamlit (推薦用於雲端部署)
1. **啟動 App**：
   ```bash
   streamlit run streamlit_app.py
   ```
2. **瀏覽網頁**：自動開啟 `http://localhost:8501`

## 雲端部署 (Streamlit Cloud)
此專案已相容於 Streamlit Cloud：
1. 將程式碼推送到 GitHub。
2. 在 [Streamlit Cloud](https://share.streamlit.io/) 點擊 "New app"。
3. 選擇此倉庫與 `streamlit_app.py`。
4. 點擊 "Deploy" 即可在線觀看！

## 數學原理
本專案使用**價值迭代 (Value Iteration)**：
\[ V_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')] \]
其中設定如下：
- 獎勵 (Reward): 到達終點 +10，每步移動 -1
- 折扣因子 (Gamma): $0.9$
- 最佳政策 ($\pi^*$): 每個狀態選擇使價值最大的行動
