# taiwan-stock-screen

## 核心定位

本系統不是選股系統。

本系統的目標是：

1. 先以 00981A 最新持股建立股票池
2. 再從股票池中尋找技術面可進場個股
3. MACD 為主要 Timing 工具
4. RS 為強弱排序工具
5. 00981A 持股變化僅作為輔助資訊

## 交易哲學

先有股票池。

再找 Timing。

不是因為 ETF 買進就買。

也不是因為 ETF 持有就買。

必須同時符合技術面條件。

## MACD 參數

- DIF = EMA(8) - EMA(17)
- MACD = DIF EMA(9)
- OSC = DIF - MACD

## A Setup

必須同時符合：

- MA20 > MA60
- Close > MA60
- OSC > 0
- OSC 今日 > OSC 昨日

說明：

代表多頭趨勢成立且動能持續擴張。

若 OSC 開始縮短，即使仍為正值，也不能列為 A。

## B Setup

必須同時符合：

- MA20 > MA60
- Close > MA60
- OSC 今日 > OSC 昨日
- OSC 翻正價格距離現價 5% 以內

說明：

代表接近 MACD 轉強點。

若 OSC 衰退，不得列入 B。

## C Setup

不符合 A 或 B。

## D Setup

- MA20 <= MA60
或
- Close <= MA60

直接淘汰。

## 重要規則

OSC Expansion 為硬條件。

系統只接受：

- Momentum Expansion

不接受：

- Momentum Deceleration

例如：

OSC 今日 < OSC 昨日

即使：

- DIF > MACD
- OSC > 0

仍不得列入 A 或 B。

## 額外輸出

系統會輸出：

- osc_flip_price
- ma20_upturn_price
- rs20_rank
- rs_accel
- osc_expanding

其中：

osc_flip_price = 明天 OSC 翻正所需價格

ma20_upturn_price = 明天 MA20 上彎所需價格

## ETF 資料

目前僅使用：

- 00981A

已移除：

- 00982A
- 00992A
- 雙 ETF 評分

## 評分邏輯

優先順序：

1. 技術面 Timing
2. RS 強弱
3. 00981A 資金流

技術面永遠優先於 ETF 持股變化。
