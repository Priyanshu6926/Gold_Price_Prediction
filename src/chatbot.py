"""
AI Quant Assistant & Financial Reasoning Chatbot for Indian Gold Markets.
Answers natural language queries about live prices, technical signals, buy/sell recommendations,
and multi-day future price predictions using live model pipelines.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
from data_loader import download_indian_market_data
from features import prepare_full_features
from forecast import forecast_future_prices


class GoldAdvisorChatbot:
    def __init__(self, raw_df: pd.DataFrame = None, feat_df: pd.DataFrame = None):
        self.raw_df = raw_df if raw_df is not None else download_indian_market_data()
        self.feat_df = feat_df if feat_df is not None else prepare_full_features(self.raw_df, save=False)

    def get_market_state(self) -> dict:
        """
        Extracts real-time prices, technical indicators, and support/resistance.
        """
        latest_raw = self.raw_df.iloc[-1]
        prev_raw = self.raw_df.iloc[-2]
        latest_feat = self.feat_df.iloc[-1]
        
        gld_bees = float(latest_raw['GOLDBEES'])
        prev_bees = float(prev_raw['GOLDBEES'])
        change_bees = gld_bees - prev_bees
        pct_bees = (change_bees / prev_bees) * 100.0
        
        gold_10g = float(latest_raw['GOLD_INR_10G']) if 'GOLD_INR_10G' in latest_raw else gld_bees * 1000
        prev_10g = float(prev_raw['GOLD_INR_10G']) if 'GOLD_INR_10G' in prev_raw else prev_bees * 1000
        change_10g = gold_10g - prev_10g
        pct_10g = (change_10g / prev_10g) * 100.0
        
        usdinr = float(latest_raw['USDINR']) if 'USDINR' in latest_raw else 85.0
        nifty = float(latest_raw['NIFTY50']) if 'NIFTY50' in latest_raw else 24000.0
        vix = float(latest_raw['INDIA_VIX']) if 'INDIA_VIX' in latest_raw else 12.0
        
        rsi = float(latest_feat['RSI_14'])
        macd = float(latest_feat['MACD'])
        macd_sig = float(latest_feat['MACD_Signal'])
        sma50 = float(latest_feat['SMA_50'])
        sma200 = float(latest_feat['SMA_200'])
        bb_upper = float(latest_feat['BB_Upper'])
        bb_lower = float(latest_feat['BB_Lower'])
        
        # 20-day Support and Resistance
        recent_20 = self.raw_df.tail(20)
        res_bees = float(recent_20['GOLDBEES'].max())
        sup_bees = float(recent_20['GOLDBEES'].min())
        res_10g = float(recent_20['GOLD_INR_10G'].max()) if 'GOLD_INR_10G' in recent_20 else res_bees * 1000
        sup_10g = float(recent_20['GOLD_INR_10G'].min()) if 'GOLD_INR_10G' in recent_20 else sup_bees * 1000
        
        # Quant Signal Decision Matrix
        score = 0
        signals_detail = []
        
        if rsi < 35:
            score += 2
            signals_detail.append("🟢 **RSI Oversold (<35)**: Potential upward mean reversion.")
        elif rsi > 70:
            score -= 2
            signals_detail.append("🔴 **RSI Overbought (>70)**: High probability of near-term consolidation or pullback.")
        else:
            signals_detail.append(f"⚪ **RSI Neutral ({rsi:.1f})**: Healthy momentum zone.")
            
        if macd > macd_sig:
            score += 1
            signals_detail.append("🟢 **MACD Bullish Cross**: Short-term momentum is positive.")
        else:
            score -= 1
            signals_detail.append("🔴 **MACD Bearish Pressure**: Short-term momentum is slowing.")
            
        if gld_bees > sma50:
            score += 1
            signals_detail.append(f"🟢 **Above 50-DMA (₹{sma50:.2f})**: Intermediate uptrend intact.")
        else:
            score -= 1
            signals_detail.append(f"🔴 **Below 50-DMA (₹{sma50:.2f})**: Intermediate trend weakness.")
            
        if sma50 > sma200:
            score += 1
            signals_detail.append("🟢 **Golden Cross**: 50-DMA is trading above 200-DMA (Macro Bullish).")
            
        if score >= 3:
            recommendation = "STRONG BUY 🟢"
            action_summary = "Technical and macro momentum strongly favor upward continuation. Ideal for adding fresh positions."
        elif score in [1, 2]:
            recommendation = "BUY ON DIPS / ACCUMULATE 🟡"
            action_summary = "Underlying trend is positive. Accumulate systematically on minor intraday or weekly pullbacks."
        elif score in [-1, 0]:
            recommendation = "HOLD / NEUTRAL ⚪"
            action_summary = "Market is in equilibrium/consolidation. Hold existing allocations without aggressive new leverage."
        else:
            recommendation = "CAUTION / PARTIAL PROFIT BOOKING 🔴"
            action_summary = "Technical indicators suggest overbought conditions or momentum exhaustion. Consider locking in short-term gains."
            
        return {
            'date': self.raw_df.index[-1].strftime('%B %d, %Y'),
            'gld_bees': gld_bees,
            'change_bees': change_bees,
            'pct_bees': pct_bees,
            'gold_10g': gold_10g,
            'change_10g': change_10g,
            'pct_10g': pct_10g,
            'usdinr': usdinr,
            'nifty': nifty,
            'vix': vix,
            'rsi': rsi,
            'macd': macd,
            'macd_sig': macd_sig,
            'sma50': sma50,
            'sma200': sma200,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'res_bees': res_bees,
            'sup_bees': sup_bees,
            'res_10g': res_10g,
            'sup_10g': sup_10g,
            'score': score,
            'recommendation': recommendation,
            'action_summary': action_summary,
            'signals_detail': signals_detail
        }

    def generate_response(self, user_query: str) -> str:
        """
        Interprets user prompt and generates detailed financial analysis.
        """
        query = user_query.lower().strip()
        state = self.get_market_state()
        
        # 1. Buy / Sell Recommendation
        if any(w in query for w in ['buy', 'sell', 'kharid', 'bech', 'recommend', 'signal', 'entry', 'exit', 'invest', 'action', 'should i']):
            signals_text = "\n".join([f"- {s}" for s in state['signals_detail']])
            return (
                f"### 🎯 Indian Gold AI Trading Signal: **{state['recommendation']}**\n\n"
                f"**Action Plan**: {state['action_summary']}\n\n"
                f"#### 📊 Technical Breakdown:\n"
                f"{signals_text}\n\n"
                f"#### 🛡️ Key Price Levels:\n"
                f"- **Immediate Support**: ₹{state['sup_bees']:.2f} (Gold BeES) / ₹{state['sup_10g']:,.0f} (10g Gold)\n"
                f"- **Immediate Resistance**: ₹{state['res_bees']:.2f} (Gold BeES) / ₹{state['res_10g']:,.0f} (10g Gold)\n\n"
                f"⚠️ *Disclaimer: Algorithmic signals are for quantitative research. Always manage risk and consult a financial advisor.*"
            )

        # 2. Future Price Predictions / Forecast
        if any(w in query for w in ['forecast', 'predict', 'future', 'tomorrow', 'next week', 'next month', 'target', 'aage', '30 day', '7 day', '14 day']):
            days = 7
            if 'month' in query or '30' in query:
                days = 30
            elif '14' in query or 'two week' in query or '2 week' in query:
                days = 14
                
            fc_df = forecast_future_prices(days_ahead=days)
            p_next = fc_df.iloc[0]
            p_final = fc_df.iloc[-1]
            
            table_rows = []
            sample_steps = [1, min(7, days), days]
            for step in sorted(list(set(sample_steps))):
                row = fc_df[fc_df['Day_Horizon'] == step].iloc[0]
                table_rows.append(
                    f"| {row['Date']} | Day +{row['Day_Horizon']} | ₹{row['Predicted_GoldBeES (₹)']:.2f} | ₹{row['Predicted_10g_24K (₹)']:,.0f} | {row['Expected_Return (%)']:+.2f}% |"
                )
            table_text = "\n".join(table_rows)
            
            return (
                f"### 🔮 AI Model Multi-Day Price Projections ({days}-Day Horizon)\n\n"
                f"Based on our **Stacking Ensemble & LightGBM** models with 95% volatility confidence intervals:\n\n"
                f"- **Next Trading Day**: **₹{p_next['Predicted_GoldBeES (₹)']:.2f}** for Gold BeES (~**₹{p_next['Predicted_10g_24K (₹)']:,.0f}** for 10g 24K Gold) [{p_next['Expected_Return (%)']:+.2f}%]\n"
                f"- **{days}-Day Target**: **₹{p_final['Predicted_GoldBeES (₹)']:.2f}** for Gold BeES (~**₹{p_final['Predicted_10g_24K (₹)']:,.0f}** for 10g 24K Gold) [{p_final['Expected_Return (%)']:+.2f}%]\n\n"
                f"| Target Date | Horizon | Gold BeES (₹) | 24K Gold (₹/10g) | Expected Return |\n"
                f"| :--- | :---: | :---: | :---: | :---: |\n"
                f"{table_text}\n\n"
                f"📈 *View the full interactive Fan Chart in the 'Future Price Forecast' tab.*"
            )

        # 3. Support & Resistance Levels
        if any(w in query for w in ['support', 'resistance', 'levels', 'range', 'stop loss']):
            return (
                f"### 🛡️ Key Indian Gold Support & Resistance Levels\n\n"
                f"#### **Nippon Gold BeES (₹ / unit)**:\n"
                f"- **Immediate Support (20D Low)**: **₹{state['sup_bees']:.2f}**\n"
                f"- **Lower Bollinger Band**: **₹{state['bb_lower']:.2f}**\n"
                f"- **Immediate Resistance (20D High)**: **₹{state['res_bees']:.2f}**\n"
                f"- **Upper Bollinger Band**: **₹{state['bb_upper']:.2f}**\n\n"
                f"#### **24K Gold (₹ / 10 Grams)**:\n"
                f"- **Immediate Support**: **₹{state['sup_10g']:,.0f}**\n"
                f"- **Immediate Resistance**: **₹{state['res_10g']:,.0f}**"
            )

        # 4. Macro Drivers (USD/INR, NIFTY, RBI)
        if any(w in query for w in ['macro', 'dollar', 'usdinr', 'rupee', 'nifty', 'inflation', 'fed', 'rbi', 'why', 'affect', 'driver']):
            return (
                f"### 🌍 Macroeconomic Drivers Influencing Indian Gold Prices\n\n"
                f"1. **USD/INR Exchange Rate (Current: ₹{state['usdinr']:.2f})**:\n"
                f"   - Since India imports over 90% of its gold, any Rupee depreciation raises domestic gold prices in INR even if global prices are stable.\n\n"
                f"2. **Domestic Equity Sentiment (NIFTY 50: {state['nifty']:,.0f})**:\n"
                f"   - When Indian equities experience volatility or correction, institutional capital rotates into Gold BeES as a hedge.\n\n"
                f"3. **India VIX ({state['vix']:.2f})**:\n"
                f"   - Lower VIX indicates market complacency, while VIX spikes (>18) trigger immediate safe-haven demand."
            )

        # 5. Gold BeES vs Physical Gold
        if any(w in query for w in ['bees', 'etf', 'physical', 'jewel', 'sovereign', 'sgb', 'tax', 'difference']):
            return (
                f"### 🪙 Gold BeES vs Physical 24K Gold vs SGB in India\n\n"
                f"1. **Nippon India ETF Gold BeES (`GOLDBEES`)**:\n"
                f"   - Traded on NSE/BSE like a stock with zero making charges.\n"
                f"   - Extremely high liquidity (instant buy/sell via Demat).\n"
                f"   - Backed 99.5%+ pure physical gold stored in secure vaults.\n\n"
                f"2. **Physical Gold (Coins/Jewelry)**:\n"
                f"   - Carries making charges (8% - 25%) and 3% GST.\n"
                f"   - Storage and purity verification friction.\n\n"
                f"3. **Key Advice**: For trading or disciplined SIP accumulation, Gold BeES is cost-efficient and tracks live spot rates tightly."
            )

        # 6. Price Inquiry
        if any(w in query for w in ['today', 'price', 'rate', 'aaj', 'cost', 'current', 'how much']):
            return (
                f"### 🪙 Live Indian Gold Price Summary ({state['date']})\n\n"
                f"- **Nippon India ETF Gold BeES (`GOLDBEES.NS`)**: **₹{state['gld_bees']:.2f}** ({state['change_bees']:+.2f} / {state['pct_bees']:+.2f}%)\n"
                f"- **Domestic Spot Gold (24K per 10 Grams)**: **₹{state['gold_10g']:,.0f}** ({state['change_10g']:+,.0f} / {state['pct_10g']:+.2f}%)\n"
                f"- **USD / INR Rate**: **₹{state['usdinr']:.2f}**\n"
                f"- **NIFTY 50**: **{state['nifty']:,.2f}** | **India VIX**: **{state['vix']:.2f}**\n\n"
                f"💡 *Gold BeES is backed 1:1 by physical 24K gold and can be traded directly on Zerodha, Groww, AngelOne, Upstox via your demat account.*"
            )

        # Default General Assistant Response
        return (
            f"### 🤖 Indian Gold AI Advisor\n\n"
            f"I can assist you with real-time quantitative insights across the Indian Gold market:\n\n"
            f"- **Today's Rates**: Gold BeES is **₹{state['gld_bees']:.2f}** | 24K Gold (10g) is **₹{state['gold_10g']:,.0f}**\n"
            f"- **Current Signal**: **{state['recommendation']}**\n"
            f"- **Ask me anything**: \n"
            f"  • *'What is today's price?'*\n"
            f"  • *'Should I buy or sell right now?'*\n"
            f"  • *'Forecast gold price for next 7 days and 30 days'*\n"
            f"  • *'What are the support and resistance levels?'*\n"
            f"  • *'How does USD/INR affect gold in India?'*"
        )
