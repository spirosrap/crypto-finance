# Bot V2 Planning Notes

## 1. What problems did V1 expose?
- Fragile edge in Weak volatility: Average TQS = 0.12
- All trades so far are in Weak regime → limited environment testing
- A few trades have very high MAE or very low efficiency → suboptimal entries or mismatched TP logic
- System logic is fixed and stable, but lacks adaptability

## 2. What features or logic do I now suspect could improve edge or robustness?
- Regime transition detection (ATR% trend, not just absolute value)
- Context-aware filters before entry (e.g. "was volatility just rising?")
- TQS pre-trade prediction using session, ATR%, trend, etc.
- Volatility-based confirmation layers (e.g. volume + ATR spike)

## 3. What complexity is worth adding, only if test results justify it?
- Real-time volatility regime shift detection
- Sizing logic based on predicted TQS tier
- Rejection logic (entry cancels if certain conditions hit)

## 4. What must stay simple or fixed in V2 to avoid collapse?
- TP/SL must remain fixed per-trade at entry — no mid-trade changes
- Core entry signal (e.g. RSI Dip or others) must be modular and testable
- Trade logic must be auditable: every decision must leave a trail
- Simplicity first: layering must follow proven utility, not complexity hunger

## 5. If V1 is "surgical but primitive," what does V2 become?
V2 is a **diagnostic surgeon**: still surgical, but informed by telemetry and context.
It balances real-time awareness (transition detection) with rule-bound clarity (locked logic).

## 6. What signals will tell me that it’s time to start V2?
- Completion of 50 trades in Live Test Cycle #1
- Detection of mixed volatility regimes in the log (Weak + Moderate + Strong)
- Emergence of repeatable underperformance patterns in specific regimes
- Clear win-rate or TQS divergence between sessions or volatility levels

---

(Notes written post-dishwashing to mark phase completion. Grounded, not reactive.)

