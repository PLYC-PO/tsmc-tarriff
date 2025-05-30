# tsmc-tarriff
## 專案目標
以事件研究法（Event Study）分析川普宣布關稅政策對台積電（TSMC, 2330.TW）股價的影響，並練習 Python 資料處理與視覺化技能。

## 動機
* 在投資學課程中接觸到「累積異常報酬（CAR）」可用於衡量事件對股價的影響，激發我以實際資料驗證理論的興趣。
* 剛學完 pandas、seaborn 等 Python 資料分析工具，希望透過專案整合並實作所學。
* 關注金融時事，因此選擇川普於 2025/4/2 宣布對中國加徵關稅作為具代表性且具體可操作的分析事件。

## 執行步驟
1. 從台灣證券交易所（TWSE）取得台積電（2330.TW）與台灣加權股價指數（TAIEX）的收盤價資料

2. 設定事件日：2025/4/7（川普於 4/2 宣布政策，台股因清明連假延後反應）

3. 定義估計期間（2024/09 至 2024/12）與事件期間（2025/4/1 至 2025/4/9）

4. 使用 Python pandas 清理並計算個股與市場的日報酬率

5. 透過 statsmodels OLS 建立市場模型，估算異常報酬（AR）與累積異常報酬（CAR）

6. 使用 seaborn 繪製散佈圖呈現分析結果

## 專案成果
1. 台積電在 4/7 當日跌停，市場預測跌幅為 14%，實際跌停限制為 10%，顯示存在顯著異常報酬。

2. 理解 CAR「相對預期」的概念，即 CAR 為正並非絕對上漲，而是相較市場預期表現較佳。

3. 熟練 pandas 處理時間序列資料，應用 OLS 模型分析，使用 seaborn 視覺化結果，完成邏輯性完整的資料分析專案。

4. 此次分析過程需手動下載多個 CSV 檔，效率不佳，因此決定未來學習 Python 爬蟲技術，以自動化資料擷取流程。
5. 因為是第一次做小專案Code太醜以後會進一步優化改善


