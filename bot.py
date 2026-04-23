"""
Polymarket SOL 5m Bot - ML Complete
=====================================
הבוט המלא עם ML.
- bootstrap.py מאמן מודל על 30 ימי היסטוריה
- הבוט מחליט לפי תחזית ML של הדקה האחרונה בלבד
- ללא סף odds קבוע — ML מחליט הכל

הרצה ראשונה:
    python bot.py --bootstrap

הרצה רגילה:
    python bot.py

Install:
    pip install py-clob-client requests python-dotenv xgboost scikit-learn pandas numpy joblib
"""

import argparse
import json
import math
import time
import sqlite3
import os
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

load_dotenv()
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# ── Settings ───────────────────────────────────────────────────────────────────
BET_SECONDS_BEFORE = 293   # start 15 seconds after window opens
RECHECK_INTERVAL   = 1     # recheck every 1 second for take profit
ML_CONF_MIN        = 0.86  # minimum ML confidence
ML_CONF_MAX        = 0.95
MAX_ODDS           = 55    # only bet when odds below 80¢
MIN_EDGE           = 0  # minimum edge (ML prob - market price) = 7%
MAX_BETS_PER_WINDOW= 1
TAKE_PROFIT_PCT    = 0.30  # sell when odds rise 30% from entry
STOP_LOSS_PCT      = 0.30  # sell when odds drop 30% from entry
WINDOW_SECONDS     = 300

def get_bet_amount(confidence):
    return 1.0  # fixed $1

# ── Trend Filter (10 min) ───────────────────────────────────────────────────────
def get_trend_10m():
    """Returns trend direction over last 10 minutes: 'Up', 'Down', or None"""
    try:
        r = requests.get(f"{BINANCE_API}/klines",
            params={"symbol": SYMBOL, "interval": "1m", "limit": 12},
            timeout=5).json()
        if len(r) < 10: return None
        closes = [float(k[4]) for k in r]
        start  = closes[-10]
        end    = closes[-1]
        change = (end - start) / start * 100
        if change > 0.1:  return "Up"
        if change < -0.1: return "Down"
        return None  # sideways — no filter
    except: return None

# ── Fair Probability (Brownian Motion) ─────────────────────────────────────────
def calc_fair_prob(ptb_price, cur_price, secs_left):
    """Calculate fair probability based on current price vs PTB and time remaining"""
    if not ptb_price or not cur_price or secs_left <= 0:
        return None
    try:
        price_change = (cur_price - ptb_price) / ptb_price  # % move from PTB
        # Brownian motion: fair_prob = 0.5 + (|Δ| / time_decay) * 5.0
        time_decay = max(secs_left, 1)
        fair_prob  = 0.5 + (price_change / time_decay) * 500
        fair_prob  = max(0.05, min(0.95, fair_prob))  # clamp
        return round(fair_prob, 3)
    except: return None

GAMMA_URL          = "https://gamma-api.polymarket.com"
CLOB_HOST          = "https://clob.polymarket.com"
CHAIN_ID           = 137
DB_PATH            = "sol_data.db"
MODEL_PATH         = "sol_model.pkl"
BINANCE_API        = "https://api.binance.com/api/v3"
SYMBOL             = "SOLUSDT"
MARKET_SLUG        = "sol-updown-5m"
DAYS_BACK          = 30

FEATURE_COLS = [
    # Volume
    "volume_1m","volume_3m","volume_5m","volume_accel",
    "buy_ratio","delta_volume","delta_accel","vol_spike",
    # Volatility
    "atr_short","atr_long","atr_ratio","volatility_expand",
    # Momentum
    "price_slope_1m","price_slope_3m","return_pressure",
    # Order book
    "bid_ask_spread","book_imbalance",
    # BTC/ETH
    "btc_slope_1m","btc_slope_3m","btc_vol_spike","sol_btc_relative",
    "eth_slope_1m","eth_slope_3m","sol_eth_relative",
    # Derivatives
    "funding_rate","oi_change","liq_proxy",
    # Time
    "hour_sin","hour_cos","minute_sin","minute_cos",
    # Price to Beat — most important!
    "dist_to_beat","beating_now","momentum_to_ptb",
]

# ── Database ───────────────────────────────────────────────────────────────────
def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS features (
        window_ts        INTEGER PRIMARY KEY,
        price_change_pct REAL, high_low_range REAL, close_location REAL,
        wick_upper REAL, wick_lower REAL, wick_imbalance REAL,
        volume_1m REAL, volume_3m REAL, volume_5m REAL, volume_accel REAL,
        buy_ratio REAL, delta_volume REAL, delta_accel REAL,
        atr_short REAL, atr_long REAL, atr_ratio REAL, volatility_expand REAL,
        price_slope_1m REAL, price_slope_3m REAL, return_pressure REAL,
        bid_ask_spread REAL, book_imbalance REAL,
        hour_sin REAL, hour_cos REAL, minute_sin REAL, minute_cos REAL,
        vol_spike REAL, btc_slope_1m REAL, btc_slope_3m REAL,
        btc_vol_spike REAL, sol_btc_relative REAL, funding_rate REAL,
        eth_slope_1m REAL, eth_slope_3m REAL, sol_eth_relative REAL,
        oi_change REAL, liq_proxy REAL,
        outcome TEXT
    )""")
    # Add missing columns if upgrading from old DB
    new_cols = ["vol_spike","btc_slope_1m","btc_slope_3m","btc_vol_spike",
                "sol_btc_relative","funding_rate","eth_slope_1m","eth_slope_3m",
                "sol_eth_relative","oi_change","liq_proxy",
                "dist_to_beat","beating_now","momentum_to_ptb"]
    for col in new_cols:
        try: db.execute(f"ALTER TABLE features ADD COLUMN {col} REAL DEFAULT 0")
        except: pass
    db.commit(); db.close()

# ── Timing ─────────────────────────────────────────────────────────────────────
def current_window_ts():
    now = int(datetime.now(timezone.utc).timestamp())
    return (now // WINDOW_SECONDS) * WINDOW_SECONDS

def seconds_until_close():
    now = int(datetime.now(timezone.utc).timestamp())
    return (current_window_ts() + WINDOW_SECONDS) - now

# ── Chainlink SOL/USD price (same source as Polymarket) ───────────────────────
_chainlink_contract = None
_chainlink_decimals = None

def get_chainlink_price():
    """Read SOL/USD price directly from Chainlink on Polygon"""
    global _chainlink_contract, _chainlink_decimals
    try:
        from web3 import Web3
        if _chainlink_contract is None:
            w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
            abi = [{"inputs":[],"name":"latestRoundData","outputs":[
                {"name":"roundId","type":"uint80"},{"name":"answer","type":"int256"},
                {"name":"startedAt","type":"uint256"},{"name":"updatedAt","type":"uint256"},
                {"name":"answeredInRound","type":"uint80"}
            ],"stateMutability":"view","type":"function"},
            {"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],
             "stateMutability":"view","type":"function"}]
            addr = Web3.to_checksum_address("0x10C8264C0935b3B9870013e057f330Ff3e9C56dC")  # SOL/USD on Polygon
            _chainlink_contract = w3.eth.contract(address=addr, abi=abi)
            _chainlink_decimals = _chainlink_contract.functions.decimals().call()

        data  = _chainlink_contract.functions.latestRoundData().call()
        price = data[1] / (10 ** _chainlink_decimals)
        return float(price)
    except:
        return None  # fallback to Binance if Chainlink fails

# ── Feature collection (last 1 minute) ────────────────────────────────────────
def collect_features(ts):
    """Collect features based on last minute of price/volume action"""
    try:
        # SOL candles
        klines = requests.get(f"{BINANCE_API}/klines",
            params={"symbol": SYMBOL, "interval": "1m", "limit": 20},
            timeout=5).json()
        if len(klines) < 5: return None

        candles = [{
            "open":    float(k[1]), "high": float(k[2]),
            "low":     float(k[3]), "close": float(k[4]),
            "volume":  float(k[5]), "buy_vol": float(k[9]),
        } for k in klines]
        for c in candles:
            c["sell_vol"] = c["volume"] - c["buy_vol"]
            c["delta"]    = c["buy_vol"] - c["sell_vol"]

        # Use candles[-2] as "last" — exclude the current opening candle to avoid leakage
        last = candles[-2]; p3 = candles[-4:-1]; p5 = candles[-6:-1]

        # Price action
        hl  = last["high"] - last["low"]
        cl  = (last["close"] - last["low"]) / hl if hl > 0 else 0.5
        wu  = (last["high"] - max(last["open"], last["close"])) / hl if hl > 0 else 0
        wl  = (min(last["open"], last["close"]) - last["low"]) / hl if hl > 0 else 0

        # Volume
        v1  = last["volume"]; v3 = sum(c["volume"] for c in p3); v5 = sum(c["volume"] for c in p5)
        vp  = sum(c["volume"] for c in candles[-6:-3]) / 3 if len(candles) >= 6 else v3/3
        va  = (v3/3 - vp) / vp * 100 if vp > 0 else 0
        bv  = sum(c["buy_vol"] for c in p5); sv = sum(c["sell_vol"] for c in p5)
        tv  = bv + sv; br = bv/tv if tv>0 else 0.5; dv = bv-sv
        dp  = sum(c["delta"] for c in candles[-10:-5]); dn_ = sum(c["delta"] for c in p5)
        da  = (dn_-dp)/abs(dp)*100 if dp!=0 else 0

        # Volume spike — is current volume much higher than average?
        avg_vol = sum(c["volume"] for c in candles[-10:]) / 10
        vol_spike = v1 / avg_vol if avg_vol > 0 else 1.0

        # ATR
        def atr(c_list):
            trs = [max(c_list[i]["high"]-c_list[i]["low"],
                       abs(c_list[i]["high"]-c_list[i-1]["close"]),
                       abs(c_list[i]["low"]-c_list[i-1]["close"]))
                   for i in range(1, len(c_list))]
            return sum(trs)/len(trs) if trs else 0

        as_ = atr(candles[-4:]); al = atr(candles[-16:])
        ar  = as_/al if al>0 else 1; ve = 1 if as_>al else -1

        # Momentum
        closes = [c["close"] for c in candles]
        sl1 = (closes[-1]-closes[-2])/closes[-2]*100 if closes[-2]>0 else 0
        sl3 = (closes[-1]-closes[-4])/closes[-4]*100 if len(closes)>=4 and closes[-4]>0 else 0
        rets= [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
        rp  = sum(rets[-5:])*100

        # Order book
        book = requests.get(f"{BINANCE_API}/depth",
            params={"symbol": SYMBOL, "limit": 10}, timeout=5).json()
        bids = book.get("bids",[]); asks = book.get("asks",[])
        if bids and asks:
            sp = (float(asks[0][0])-float(bids[0][0]))/float(bids[0][0])*100
            bd = sum(float(b[1]) for b in bids[:5]); ad = sum(float(a[1]) for a in asks[:5])
            bi = bd/(bd+ad) if (bd+ad)>0 else 0.5
        else: sp=0; bi=0.5

        # BTC features — SOL follows BTC ~80% of the time
        btc_sl1 = btc_sl3 = btc_vol_spike = 0.0
        sol_btc_relative = 0.0
        try:
            btc_k = requests.get(f"{BINANCE_API}/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": 10},
                timeout=5).json()
            if len(btc_k) >= 4:
                btc_closes = [float(k[4]) for k in btc_k]
                btc_vols   = [float(k[5]) for k in btc_k]
                btc_sl1 = (btc_closes[-1]-btc_closes[-2])/btc_closes[-2]*100 if btc_closes[-2]>0 else 0
                btc_sl3 = (btc_closes[-1]-btc_closes[-4])/btc_closes[-4]*100 if btc_closes[-4]>0 else 0
                avg_btc_vol = sum(btc_vols[-5:]) / 5
                btc_vol_spike = btc_vols[-1] / avg_btc_vol if avg_btc_vol > 0 else 1.0
                sol_btc_relative = sl1 - btc_sl1
        except: pass

        # ETH features
        eth_sl1 = eth_sl3 = 0.0
        sol_eth_relative = 0.0
        try:
            eth_k = requests.get(f"{BINANCE_API}/klines",
                params={"symbol": "ETHUSDT", "interval": "1m", "limit": 10},
                timeout=5).json()
            if len(eth_k) >= 4:
                eth_closes = [float(k[4]) for k in eth_k]
                eth_sl1 = (eth_closes[-1]-eth_closes[-2])/eth_closes[-2]*100 if eth_closes[-2]>0 else 0
                eth_sl3 = (eth_closes[-1]-eth_closes[-4])/eth_closes[-4]*100 if eth_closes[-4]>0 else 0
                sol_eth_relative = sl1 - eth_sl1
        except: pass

        # Open Interest change — new positions = stronger move
        oi_change = 0.0
        try:
            oi = requests.get("https://fapi.binance.com/fapi/v1/openInterestHist",
                params={"symbol": "SOLUSDT", "period": "5m", "limit": 3},
                timeout=3).json()
            if len(oi) >= 2:
                oi_now  = float(oi[-1]["sumOpenInterest"])
                oi_prev = float(oi[-2]["sumOpenInterest"])
                oi_change = (oi_now - oi_prev) / oi_prev * 100 if oi_prev > 0 else 0
        except: pass

        # Liquidations proxy — large OI drop = liquidations happened
        liq_proxy = min(oi_change, 0)  # negative OI change = longs liquidated

        # Funding rate
        funding_rate = 0.0
        try:
            fr = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": "SOLUSDT", "limit": 1}, timeout=3).json()
            if fr: funding_rate = float(fr[-1].get("fundingRate", 0)) * 100
        except: pass

        # Time (cyclical)
        now_ = datetime.now(timezone.utc)
        hs = math.sin(2*math.pi*now_.hour/24);   hc = math.cos(2*math.pi*now_.hour/24)
        ms_= math.sin(2*math.pi*now_.minute/60); mc = math.cos(2*math.pi*now_.minute/60)

        # Price to Beat — read directly from Chainlink (same source as Polymarket)
        price_to_beat   = get_chainlink_price()
        current_price   = float(candles[-1]["close"])
        if price_to_beat is None:
            price_to_beat = float(candles[-1]["open"])  # fallback to Binance open
        dist_to_beat    = (current_price - price_to_beat) / price_to_beat * 100
        beating_now     = 1.0 if current_price >= price_to_beat else -1.0
        prev_dist       = (float(candles[-2]["close"]) - price_to_beat) / price_to_beat * 100
        momentum_to_ptb = dist_to_beat - prev_dist

        return {
            "window_ts":        ts,
            # Volume — from previous candles only
            "volume_1m":        v1,  "volume_3m":         v3,  "volume_5m":      v5,
            "volume_accel":     va,  "buy_ratio":         br,  "delta_volume":   dv,
            "delta_accel":      da,
            # Volatility — from previous candles only
            "atr_short":        as_, "atr_long":          al,
            "atr_ratio":        ar,  "volatility_expand": ve,
            # Momentum — from previous candles only
            "price_slope_1m":   sl1, "price_slope_3m":    sl3, "return_pressure":rp,
            # Order book
            "bid_ask_spread":   sp,  "book_imbalance":    bi,
            # Time
            "hour_sin":         hs,  "hour_cos":          hc,
            "minute_sin":       ms_, "minute_cos":        mc,
            # BTC/ETH/derivatives
            "vol_spike":        vol_spike,
            "btc_slope_1m":     btc_sl1,  "btc_slope_3m":  btc_sl3,
            "btc_vol_spike":    btc_vol_spike,
            "sol_btc_relative": sol_btc_relative,
            "eth_slope_1m":     eth_sl1,  "eth_slope_3m":  eth_sl3,
            "sol_eth_relative": sol_eth_relative,
            "oi_change":        oi_change,
            "liq_proxy":        liq_proxy,
            "funding_rate":     funding_rate,
            # Price to Beat — most important for Polymarket
            "dist_to_beat":     dist_to_beat,
            "beating_now":      beating_now,
            "momentum_to_ptb":  momentum_to_ptb,
            "outcome":          None,
        }
    except Exception as e:
        print(f"  Feature error: {e}"); return None

def save_features(f):
    db   = sqlite3.connect(DB_PATH)
    cols = ", ".join(f.keys())
    vals = ", ".join(["?"]*len(f))
    db.execute(f"INSERT OR REPLACE INTO features ({cols}) VALUES ({vals})", list(f.values()))
    db.commit(); db.close()

def label_prev_window(prev_ts):
    try:
        r = requests.get(f"{BINANCE_API}/klines",
            params={"symbol": SYMBOL, "interval": "5m",
                    "startTime": prev_ts*1000, "limit": 1}, timeout=5).json()
        if r:
            outcome = "Up" if float(r[0][4]) >= float(r[0][1]) else "Down"
            db = sqlite3.connect(DB_PATH)
            db.execute("UPDATE features SET outcome=? WHERE window_ts=? AND outcome IS NULL",
                       (outcome, prev_ts))
            db.commit(); db.close()
            return outcome
    except: pass
    return None

# ── ML Prediction ──────────────────────────────────────────────────────────────
def ml_predict(features):
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        saved   = joblib.load(MODEL_PATH)
        model   = saved["model"]
        X       = pd.DataFrame([{c: features.get(c, 0) for c in FEATURE_COLS}]).fillna(0)
        prob_up = model.predict_proba(X)[0][1]
        direction  = "Up" if prob_up >= 0.5 else "Down"
        confidence = prob_up if prob_up >= 0.5 else 1 - prob_up
        return {"direction": direction, "confidence": round(confidence, 3),
                "prob_up": round(prob_up, 3)}
    except Exception as e:
        print(f"  ML error: {e}"); return None

# ── Bootstrap (historical training) ───────────────────────────────────────────
def download_candles(symbol, days, binance_api):
    """Download historical 1m candles for a symbol"""
    end_ts   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 3600 * 1000)
    all_raw  = []
    current  = start_ts
    while current < end_ts:
        try:
            r = requests.get(f"{binance_api}/klines", params={
                "symbol": symbol, "interval": "1m",
                "startTime": current, "endTime": min(current+1000*60*1000, end_ts),
                "limit": 1000
            }, timeout=10).json()
            if not r: break
            all_raw.extend(r)
            current = r[-1][0] + 60000
            print(f"  {symbol}: {len(all_raw)} candles...", end="\r")
            time.sleep(0.08)
        except Exception as e:
            print(f"\n  Error: {e}"); time.sleep(1)
    print(f"\n  {symbol}: {len(all_raw)} candles total")
    return all_raw

def parse_candles(raw):
    df = pd.DataFrame(raw, columns=[
        "ts","open","high","low","close","volume",
        "close_ts","quote_vol","trades","buy_vol","buy_quote_vol","ignore"
    ])
    for c in ["open","high","low","close","volume","buy_vol"]:
        df[c] = df[c].astype(float)
    df["sell_vol"] = df["volume"] - df["buy_vol"]
    df["delta"]    = df["buy_vol"] - df["sell_vol"]
    df["ts"]       = df["ts"].astype(int) // 1000
    return df.set_index("ts").sort_index()

def bootstrap():
    print("=" * 55)
    print(f"Bootstrap: downloading {DAYS_BACK} days — SOL + BTC + ETH")
    print("=" * 55)

    sol_df = parse_candles(download_candles("SOLUSDT", DAYS_BACK, BINANCE_API))
    btc_df = parse_candles(download_candles("BTCUSDT", DAYS_BACK, BINANCE_API))
    eth_df = parse_candles(download_candles("ETHUSDT", DAYS_BACK, BINANCE_API))

    # Build features for each 5m window
    print("\nBuilding features for each 5m window...")
    now_ts  = int(datetime.now(timezone.utc).timestamp())
    start_w = ((now_ts - DAYS_BACK*24*3600) // 300) * 300
    features_list = []
    ts = start_w
    n  = 0

    while ts < now_ts - 300:
        window_end = ts + 300
        # IMPORTANT: use only candles BEFORE window open to avoid leakage
        mask   = (sol_df.index >= ts - 20*60) & (sol_df.index < ts)
        candles= sol_df[mask].tail(20)
        w_mask = (sol_df.index >= ts) & (sol_df.index < window_end)
        w_cands= sol_df[w_mask]

        if len(candles) >= 5 and len(w_cands) >= 1:
            last = candles.iloc[-1]
            p3   = candles.tail(3); p5 = candles.tail(5)

            hl  = last["high"]-last["low"]
            cl  = (last["close"]-last["low"])/hl if hl>0 else 0.5
            wu  = (last["high"]-max(last["open"],last["close"]))/hl if hl>0 else 0
            wl  = (min(last["open"],last["close"])-last["low"])/hl if hl>0 else 0

            v1  = last["volume"]; v3=p3["volume"].sum(); v5=p5["volume"].sum()
            vp  = candles.iloc[-6:-3]["volume"].mean() if len(candles)>=6 else v3/3
            va  = (v3/3-vp)/vp*100 if vp>0 else 0
            bv  = p5["buy_vol"].sum(); sv=p5["sell_vol"].sum(); tv=bv+sv
            br  = bv/tv if tv>0 else 0.5; dv=bv-sv
            dp  = candles.iloc[-10:-5]["delta"].sum()
            dn_ = p5["delta"].sum()
            da  = (dn_-dp)/abs(dp)*100 if dp!=0 else 0
            avg_vol = candles.tail(10)["volume"].mean()
            vol_spike = v1/avg_vol if avg_vol>0 else 1.0

            def atr(c):
                if len(c)<2: return 0
                trs=[max(c.iloc[i]["high"]-c.iloc[i]["low"],
                         abs(c.iloc[i]["high"]-c.iloc[i-1]["close"]),
                         abs(c.iloc[i]["low"]-c.iloc[i-1]["close"]))
                     for i in range(1,len(c))]
                return sum(trs)/len(trs)

            as_=atr(candles.tail(4)); al=atr(candles.tail(16))
            ar=as_/al if al>0 else 1; ve=1 if as_>al else -1

            closes=candles["close"].tolist()
            sl1=(closes[-1]-closes[-2])/closes[-2]*100 if closes[-2]>0 else 0
            sl3=(closes[-1]-closes[-4])/closes[-4]*100 if len(closes)>=4 and closes[-4]>0 else 0
            rets=[(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
            rp=sum(rets[-5:])*100

            dt=datetime.fromtimestamp(ts,tz=timezone.utc)
            hs=math.sin(2*math.pi*dt.hour/24); hc=math.cos(2*math.pi*dt.hour/24)
            ms_=math.sin(2*math.pi*dt.minute/60); mc=math.cos(2*math.pi*dt.minute/60)

            outcome="Up" if w_cands.iloc[-1]["close"]>=w_cands.iloc[0]["open"] else "Down"

            # BTC features
            btc_sl1=btc_sl3=btc_vol_spike=sol_btc_relative=0.0
            btc_mask = (btc_df.index >= ts-10*60) & (btc_df.index < window_end)
            btc_c    = btc_df[btc_mask].tail(10)
            if len(btc_c) >= 4:
                bc = btc_c["close"].tolist()
                bv_ = btc_c["volume"].tolist()
                btc_sl1 = (bc[-1]-bc[-2])/bc[-2]*100 if bc[-2]>0 else 0
                btc_sl3 = (bc[-1]-bc[-4])/bc[-4]*100 if bc[-4]>0 else 0
                avg_bv  = sum(bv_[-5:])/5
                btc_vol_spike = bv_[-1]/avg_bv if avg_bv>0 else 1.0
                sol_btc_relative = sl1 - btc_sl1

            # ETH features
            eth_sl1=eth_sl3=sol_eth_relative=0.0
            eth_mask = (eth_df.index >= ts-10*60) & (eth_df.index < window_end)
            eth_c    = eth_df[eth_mask].tail(10)
            if len(eth_c) >= 4:
                ec = eth_c["close"].tolist()
                eth_sl1 = (ec[-1]-ec[-2])/ec[-2]*100 if ec[-2]>0 else 0
                eth_sl3 = (ec[-1]-ec[-4])/ec[-4]*100 if ec[-4]>0 else 0
                sol_eth_relative = sl1 - eth_sl1

            features_list.append({
                "window_ts":ts,
                "volume_1m":v1,"volume_3m":v3,"volume_5m":v5,"volume_accel":va,
                "buy_ratio":br,"delta_volume":dv,"delta_accel":da,
                "atr_short":as_,"atr_long":al,"atr_ratio":ar,"volatility_expand":ve,
                "price_slope_1m":sl1,"price_slope_3m":sl3,"return_pressure":rp,
                "bid_ask_spread":hl/last["close"]*100*0.1,"book_imbalance":br,
                "hour_sin":hs,"hour_cos":hc,"minute_sin":ms_,"minute_cos":mc,
                "vol_spike":vol_spike,
                "btc_slope_1m":btc_sl1,"btc_slope_3m":btc_sl3,
                "btc_vol_spike":btc_vol_spike,"sol_btc_relative":sol_btc_relative,
                "funding_rate":0.0,
                "eth_slope_1m":eth_sl1,"eth_slope_3m":eth_sl3,
                "sol_eth_relative":sol_eth_relative,
                "oi_change":0.0,"liq_proxy":0.0,
                # Price to Beat — open of window vs close of first candle
                "dist_to_beat":    (last["close"] - w_cands.iloc[0]["open"]) / w_cands.iloc[0]["open"] * 100,
                "beating_now":     1.0 if last["close"] >= w_cands.iloc[0]["open"] else -1.0,
                "momentum_to_ptb": sl1,  # price slope as proxy for momentum toward PTB
                "outcome":outcome,
            })
        ts  += 300
        n   += 1
        if n % 200 == 0:
            print(f"  {n} windows processed, {len(features_list)} valid...", end="\r")

    print(f"\n  Built {len(features_list)} feature rows")

    # Save
    db = sqlite3.connect(DB_PATH)
    for f in features_list:
        cols=", ".join(f.keys()); vals=", ".join(["?"]*len(f))
        db.execute(f"INSERT OR REPLACE INTO features ({cols}) VALUES ({vals})", list(f.values()))
    db.commit(); db.close()

    # Train
    train_model()

def train_model():
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier

    db = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM features WHERE outcome IS NOT NULL ORDER BY window_ts", db)
    db.close()

    if len(df) < 100:
        print(f"  Not enough data ({len(df)} rows)"); return

    # Walk-forward: train only on last 14 days
    now_ts   = int(datetime.now(timezone.utc).timestamp())
    cutoff   = now_ts - 14*24*3600
    df_recent = df[df["window_ts"] >= cutoff]
    if len(df_recent) >= 200:
        df = df_recent
        print(f"\nTraining on {len(df)} samples (last 14 days)...")
    else:
        print(f"\nTraining on {len(df)} samples (all history)...")

    X = df[FEATURE_COLS].fillna(0)
    y = (df["outcome"]=="Up").astype(int)

    split = int(len(X)*0.8)

    # Ensemble: XGBoost + RandomForest
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        use_label_encoder=False, eval_metric="logloss", random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            VotingClassifier([("xgb", xgb), ("rf", rf)], voting="soft"),
            cv=3, method="isotonic"
        ))
    ])
    model.fit(X.iloc[:split], y.iloc[:split])

    probs     = model.predict_proba(X.iloc[split:])[:,1]
    preds     = (probs >= 0.5).astype(int)
    conf_mask = (probs >= ML_CONF_MIN) | (probs <= 1-ML_CONF_MIN)
    acc_all   = accuracy_score(y.iloc[split:], preds)
    acc_conf  = accuracy_score(y.iloc[split:][conf_mask], preds[conf_mask]) if conf_mask.sum()>0 else 0

    print(f"  Accuracy (all):       {acc_all:.2%}")
    print(f"  Accuracy (confident): {acc_conf:.2%} on {conf_mask.sum()}/{len(X.iloc[split:])} preds")

    # Calibration check — how accurate are "80%+" predictions?
    high_conf = (probs >= 0.80) | (probs <= 0.20)
    if high_conf.sum() > 0:
        cal_acc = accuracy_score(y.iloc[split:][high_conf], preds[high_conf])
        print(f"  Calibration (80%+):   {cal_acc:.2%} on {high_conf.sum()} preds")

    # Feature importance (from XGBoost inside ensemble)
    try:
        xgb_fitted = model.named_steps["clf"].estimator.estimators_[0]
        importances = xgb_fitted.feature_importances_
        top5 = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top features: {', '.join(f'{n}({v:.3f})' for n,v in top5)}")
    except: pass

    # Volatility hours analysis — which hours have best win rate
    try:
        df_test = df.iloc[split:].copy()
        df_test["pred"] = preds
        df_test["correct"] = (df_test["pred"] == y.iloc[split:].values).astype(int)
        df_test["hour"] = pd.to_datetime(df_test["window_ts"], unit="s").dt.hour
        hour_acc = df_test.groupby("hour")["correct"].mean()
        best_hours  = hour_acc[hour_acc >= 0.60].index.tolist()
        worst_hours = hour_acc[hour_acc < 0.45].index.tolist()
        if best_hours:  print(f"  Best hours (UTC):  {best_hours}")
        if worst_hours: print(f"  Worst hours (UTC): {worst_hours}")
        # Save to model
        joblib.dump({
            "model": model, "trained_on": len(df),
            "trained_at": datetime.now().isoformat(),
            "best_hours": best_hours, "worst_hours": worst_hours,
            "acc_all": acc_all, "acc_conf": acc_conf,
        }, MODEL_PATH)
    except:
        joblib.dump({"model": model, "trained_on": len(df),
                     "trained_at": datetime.now().isoformat()}, MODEL_PATH)

    print(f"  Model saved → {MODEL_PATH} ✓")


# ── Polymarket client ──────────────────────────────────────────────────────────
def init_client():
    client = ClobClient(
        host=CLOB_HOST, key=PRIVATE_KEY, chain_id=CHAIN_ID,
        signature_type=1, funder="0x2451262228D9988a3F5F60EEC86062EAb59B09fA",
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    print(f"[+] Connected | {client.get_address()}")
    return client

def find_market(ts):
    slug = f"{MARKET_SLUG}-{ts}"
    try:
        r = requests.get(f"{GAMMA_URL}/events", params={"slug": slug}, timeout=5).json()
        if isinstance(r, dict): r = r.get("data", [])
        if r:
            m = r[0].get("markets", [])
            if m: return m
        r2 = requests.get(f"{GAMMA_URL}/markets", params={"event_slug": slug}, timeout=5).json()
        if isinstance(r2, dict): r2 = r2.get("data", [])
        return r2 if r2 else None
    except: return None

def get_odds(client, markets):
    try:
        m         = markets[0]
        outcomes  = json.loads(m.get("outcomes", '["Up","Down"]'))
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        result    = {}
        for outcome, token_id in zip(outcomes, token_ids):
            try:
                mid   = client.get_midpoint(token_id)
                cents = round(float(mid["mid"])*100)
                low   = outcome.lower()
                if "up"   in low: result["Up"]   = {"cents": cents, "token_id": token_id}
                elif "down" in low: result["Down"] = {"cents": cents, "token_id": token_id}
            except: pass
        return result if "Up" in result and "Down" in result else None
    except: return None

def get_token_balance(client, token_id):
    """Get actual token balance from Polymarket"""
    try:
        resp = requests.get(
            f"{CLOB_HOST}/balance-allowance",
            params={"asset_type": "CONDITIONAL", "token_id": token_id},
            headers={"POLY_ADDRESS": client.get_address()},
            timeout=5
        ).json()
        balance = float(resp.get("balance", 0)) / 1e6  # convert from wei
        return balance if balance > 0 else None
    except Exception as e:
        return None

def sell_position(client, token_id, amount):
    """Sell entire position — retry until all sold"""
    try:
        from py_clob_client.order_builder.constants import SELL
        import re
        remaining = math.floor(amount * 1e6) / 1e6
        for attempt in range(10):
            if remaining <= 0.001:
                print(f"  ✅ Position fully sold")
                return True
            try:
                sell_amt = math.floor(remaining * 1e6) / 1e6
                signed = client.create_market_order(
                    MarketOrderArgs(token_id=token_id, amount=sell_amt, side=SELL, order_type=OrderType.FOK))
                resp = client.post_order(signed, OrderType.FOK)
                print(f"  💰 SELL {sell_amt:.4f}: {resp.get('status')} takingAmount={resp.get('takingAmount',0)}")
                if resp.get("success"):
                    sold = float(resp.get("makingAmount", sell_amt))
                    remaining -= sold
                    remaining = max(0, math.floor(remaining * 1e6) / 1e6)
                    if remaining > 0.001:
                        print(f"  Still {remaining:.4f} remaining — retrying...")
                        time.sleep(1); continue
                    return True
                # Parse actual balance from error
                err = str(resp.get("errorMsg","") or resp.get("error",""))
                if "balance:" in err:
                    m = re.search(r'balance: (\d+)', err)
                    if m:
                        remaining = int(m.group(1)) / 1e6
                        print(f"  Balance says {remaining:.4f} — retrying...")
                        time.sleep(1); continue
                return False
            except Exception as e:
                err = str(e)
                if "balance:" in err:
                    m = re.search(r'balance: (\d+)', err)
                    if m:
                        remaining = int(m.group(1)) / 1e6
                        time.sleep(1); continue
                if "FOK" in err or "fully filled" in err:
                    time.sleep(1); continue
                print(f"  Sell error: {e}")
                return False
        return False
    except Exception as e:
        print(f"  Sell error: {e}"); return False

def log_bet(direction, amount, odds, ml_conf, status, order_id=""):
    try:
        db = sqlite3.connect(DB_PATH)
        db.execute("INSERT INTO bets (direction,amount,odds,ml_conf,status,order_id) VALUES (?,?,?,?,?,?)",
                   (direction, amount, odds, ml_conf, status, order_id))
        db.commit(); db.close()
    except: pass

def place_bet(client, token_id, amount, direction="", odds=0, ml_conf=0):
    """Place bet — retry until fully filled, accumulate all shares"""
    import re
    total_shares = 0.0
    remaining    = math.floor(amount * 1e6) / 1e6
    for attempt in range(10):
        if remaining <= 0.001:
            log_bet(direction, amount, odds, ml_conf, "win", "")
            return True, total_shares
        try:
            signed = client.create_market_order(
                MarketOrderArgs(token_id=token_id, amount=remaining, side=BUY, order_type=OrderType.FOK))
            resp = client.post_order(signed, OrderType.FOK)
            print(f"  Order: {resp}")
            if resp.get("success"):
                shares = float(resp.get("takingAmount", remaining))
                spent  = float(resp.get("makingAmount", remaining))
                total_shares += shares
                remaining    -= spent
                remaining     = max(0, math.floor(remaining * 1e6) / 1e6)
                if remaining > 0.001:
                    print(f"  Partial fill — {remaining:.4f} remaining, {total_shares:.4f} shares so far")
                    time.sleep(1); continue
                log_bet(direction, amount, odds, ml_conf, "win", resp.get("orderID",""))
                return True, total_shares
            err = resp.get("errorMsg","") or str(resp.get("error",""))
            if "fully filled" in err or "FOK" in err:
                print(f"  FOK failed (attempt {attempt+1}/10) — retrying in 1s...")
                time.sleep(1); continue
            log_bet(direction, amount, odds, ml_conf, "failed", str(err))
            return False, 0
        except Exception as e:
            if "fully filled" in str(e) or "FOK" in str(e):
                print(f"  FOK failed (attempt {attempt+1}/10) — retrying in 1s...")
                time.sleep(1); continue
            print(f"  Error: {e}")
            log_bet(direction, amount, odds, ml_conf, "error", str(e))
            return False, 0
    if total_shares > 0:
        return True, total_shares
    log_bet(direction, amount, odds, ml_conf, "failed_all", "")
    print(f"  → All attempts failed")
    return False, 0

# ── Main bot loop ──────────────────────────────────────────────────────────────
def run_bot():
    if not PRIVATE_KEY:
        print("ERROR: Set PRIVATE_KEY in .env"); return
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found. Run first: python bot.py --bootstrap"); return

    saved = joblib.load(MODEL_PATH)
    print("=" * 55)
    print("Polymarket SOL 5m Bot - ML Edition")
    print(f"ML conf >= {ML_CONF_MIN} | max {MAX_BETS_PER_WINDOW} bets/window | dynamic sizing $2-$6")
    print(f"Model trained on {saved.get('trained_on',0)} samples | {saved.get('trained_at','')[:10]}")
    print("=" * 55)

    client       = init_client()
    placed        = set()
    window_bets   = {}
    ml_decisions  = {}
    last_check    = {}
    current_ts    = None
    markets       = None
    wins_since_retrain = 0
    # Enhanced tracking
    consecutive_losses = 0
    consecutive_wins   = 0
    stop_until         = 0
    # Take profit tracking: ts -> {"entry_odds", "direction", "token_id", "amount"}
    open_positions = {}

    while True:
        try:
            secs_left = seconds_until_close()
            ts        = current_window_ts()
            now_str   = datetime.now().strftime("%H:%M:%S")

            # ── Take Profit + Stop Loss check ─────────────────────────────────
            for pos_ts, pos in list(open_positions.items()):
                if pos_ts != ts: continue
                if pos.get("sold"): continue
                try:
                    pos_odds = get_odds(client, markets) if markets else None
                    if pos_odds and pos["direction"] in pos_odds:
                        current_cents = pos_odds[pos["direction"]]["cents"]
                        entry_cents   = pos["entry_odds"]
                        gain_pct      = (current_cents - entry_cents) / entry_cents

                        if gain_pct >= TAKE_PROFIT_PCT:
                            print(f"[{now_str}] 💰 TAKE PROFIT! {pos['direction']} {entry_cents}¢ → {current_cents}¢ (+{gain_pct:.0%})")
                            success = sell_position(client, pos["token_id"], pos["amount"])
                            if success:
                                open_positions[pos_ts]["sold"] = True
                                consecutive_wins += 1
                                consecutive_losses = 0
                                print(f"  ✅ Sold for profit!")

                        elif gain_pct <= -STOP_LOSS_PCT:
                            print(f"[{now_str}] 🛑 STOP LOSS! {pos['direction']} {entry_cents}¢ → {current_cents}¢ ({gain_pct:.0%})")
                            success = sell_position(client, pos["token_id"], pos["amount"])
                            if success:
                                open_positions[pos_ts]["sold"] = True
                                consecutive_losses += 1
                                consecutive_wins = 0
                                print(f"  ❌ Stop loss executed!")
                                if consecutive_losses >= 3:
                                    stop_until = int(time.time()) + 3600
                                    print(f"  🛑 3 losses — stopping for 1 hour")
                except Exception as e:
                    print(f"  TP/SL check error: {e}")

            if ts != current_ts:
                print(f"\n[{now_str}] New window | closes in {secs_left}s")
                markets    = find_market(ts)
                current_ts = ts

                # Label previous window
                prev_ts = ts - WINDOW_SECONDS
                outcome = label_prev_window(prev_ts)
                if outcome:
                    print(f"  [data] {prev_ts} → {outcome}")

                # Retrain every 100 windows
                wins_since_retrain += 1
                if wins_since_retrain >= 100:
                    print("  [ML] Retraining...")
                    train_model()
                    wins_since_retrain = 0

            if secs_left <= BET_SECONDS_BEFORE and ts not in placed and markets:
                now_epoch = int(time.time())

                # ── STEP 1: ML decides direction ONCE at window open ──────────
                if ts not in ml_decisions:
                    features = collect_features(ts)
                    if features:
                        save_features(features)
                    pred = ml_predict(features) if features else None
                    if not pred:
                        print(f"[{now_str}] ML unavailable — skipping window")
                        placed.add(ts); time.sleep(1); continue

                    ml_decisions[ts] = {"direction": pred["direction"], "confidence": pred["confidence"]}
                    print(f"[{now_str}] ✓ ML locked: {pred['direction']} ({pred['confidence']:.0%}) — watching PTB...")

                ml_dir  = ml_decisions[ts]["direction"]
                ml_conf = ml_decisions[ts]["confidence"]

                # ── STEP 2: Check PTB every 20s for timing ───────────────────
                if ts in last_check and (now_epoch - last_check[ts]) < RECHECK_INTERVAL:
                    time.sleep(1); continue
                last_check[ts] = now_epoch

                fresh = find_market(ts)
                if fresh: markets = fresh

                odds = get_odds(client, markets)
                if not odds:
                    print(f"[{now_str}] Could not read odds")
                    time.sleep(1); continue

                up_c = odds["Up"]["cents"]; dn_c = odds["Down"]["cents"]

                # Get current PTB status from Chainlink
                cl_price = get_chainlink_price()
                ptb_price = None
                try:
                    sol_k = requests.get(f"{BINANCE_API}/klines",
                        params={"symbol": SYMBOL, "interval": "5m", "limit": 1},
                        timeout=5).json()
                    ptb_price = float(sol_k[0][1]) if sol_k else None  # open of current 5m candle
                except: pass

                cur_price = None
                try:
                    t = requests.get(f"{BINANCE_API}/ticker/price",
                        params={"symbol": SYMBOL}, timeout=3).json()
                    cur_price = float(t["price"])
                except: pass

                # PTB confirmation logic
                ptb_confirm = False
                ptb_status  = "unknown"

                if ptb_price and cur_price:
                    beating     = cur_price >= ptb_price
                    dist_pct    = (cur_price - ptb_price) / ptb_price * 100

                    if ml_dir == "Up":
                        # For Up: price should be above PTB or moving toward it
                        if beating:
                            ptb_confirm = True
                            ptb_status  = f"✓ beating PTB ${ptb_price:.2f} by {dist_pct:+.2f}%"
                        elif dist_pct > -0.3:
                            # Very close to PTB — still good entry
                            ptb_confirm = True
                            ptb_status  = f"≈ near PTB ${ptb_price:.2f} ({dist_pct:+.2f}%)"
                        else:
                            ptb_status  = f"✗ below PTB ${ptb_price:.2f} ({dist_pct:+.2f}%)"

                    elif ml_dir == "Down":
                        # For Down: price should be below PTB
                        if not beating:
                            ptb_confirm = True
                            ptb_status  = f"✓ below PTB ${ptb_price:.2f} ({dist_pct:+.2f}%)"
                        elif dist_pct < 0.3:
                            ptb_confirm = True
                            ptb_status  = f"≈ near PTB ${ptb_price:.2f} ({dist_pct:+.2f}%)"
                        else:
                            ptb_status  = f"✗ above PTB ${ptb_price:.2f} ({dist_pct:+.2f}%)"
                else:
                    # No PTB data — fallback to ML only
                    ptb_confirm = True
                    ptb_status  = "PTB unavailable — ML only"

                print(f"[{now_str}] Up:{up_c}¢ Down:{dn_c}¢ | ML={ml_dir} {ml_conf:.0%} | PTB: {ptb_status} | {secs_left}s")

                # Skip if market resolved
                if up_c <= 1 or dn_c <= 1:
                    print(f"  → Market resolved — skip")
                    placed.add(ts)
                    time.sleep(1); continue

                # ── Stop-loss check ──────────────────────────────────────────
                now_epoch2 = int(time.time())
                if now_epoch2 < stop_until:
                    remaining = stop_until - now_epoch2
                    print(f"[{now_str}] 🛑 Stop-loss active — resuming in {remaining//60}m {remaining%60}s")
                    time.sleep(30); continue

                # ── Hour filter — skip bad hours ──────────────────────────────
                saved_model = joblib.load(MODEL_PATH)
                worst_hours = saved_model.get("worst_hours", [])
                cur_hour    = datetime.now(timezone.utc).hour
                if cur_hour in worst_hours:
                    print(f"[{now_str}] ⏰ Hour {cur_hour} UTC has low win rate — skipping")
                    placed.add(ts); time.sleep(1); continue

                # ── Volatility filter ─────────────────────────────────────────
                try:
                    sol_atr = requests.get(f"{BINANCE_API}/klines",
                        params={"symbol": SYMBOL, "interval": "1m", "limit": 20}, timeout=5).json()
                    highs  = [float(k[2]) for k in sol_atr]
                    lows   = [float(k[3]) for k in sol_atr]
                    closes = [float(k[4]) for k in sol_atr]
                    trs    = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
                              for i in range(1, len(closes))]
                    atr_now = sum(trs[-5:])/5
                    atr_avg = sum(trs)/len(trs)
                    if atr_now > atr_avg * 1.8:
                        print(f"[{now_str}] ⚡ Volatility too high (ATR {atr_now:.3f} > {atr_avg*1.8:.3f}) — skipping")
                        time.sleep(1); continue
                except: pass

                # ── Anti-streak sizing ────────────────────────────────────────
                streak_multiplier = 1.0
                if consecutive_wins >= 3:
                    streak_multiplier = 1.5
                    print(f"  🔥 {consecutive_wins} wins streak — sizing up x1.5")

                # BET only when ML + PTB both confirm
                if ptb_confirm:
                    bets_this_window = window_bets.get(ts, 0)
                    if bets_this_window >= MAX_BETS_PER_WINDOW:
                        print(f"  → MAX {MAX_BETS_PER_WINDOW} bets reached this window — skip")
                        placed.add(ts); time.sleep(1); continue

                    current_odds = odds[ml_dir]["cents"]
                    if current_odds > MAX_ODDS:
                        print(f"  → SKIP — {ml_dir} at {current_odds}¢ > {MAX_ODDS}¢")
                        time.sleep(1); continue

                    # Edge calculation
                    market_price = current_odds / 100
                    edge = ml_conf - market_price
                    if edge < MIN_EDGE:
                        print(f"  → SKIP — Edge {edge:.2%} < {MIN_EDGE:.0%} (ML={ml_conf:.0%} odds={current_odds}¢)")
                        time.sleep(1); continue

                    bet_amount = round(get_bet_amount(ml_conf) * streak_multiplier, 1)
                    token_id   = odds[ml_dir]["token_id"]
                    print(f"  → BETTING ${bet_amount} on {ml_dir} at {current_odds}¢ (edge={edge:.0%} conf={ml_conf:.0%})")
                    success, shares = place_bet(client, token_id, bet_amount, ml_dir, current_odds, ml_conf)
                    window_bets[ts] = bets_this_window + 1
                    placed.add(ts)

                    if success:
                        consecutive_wins  += 1
                        consecutive_losses = 0
                        time.sleep(5)
                        # Round to 6 decimal places (Polymarket precision)
                        safe_shares = math.floor(shares * 1e6) / 1e6
                        open_positions[ts] = {
                            "direction":  ml_dir,
                            "entry_odds": current_odds,
                            "token_id":   token_id,
                            "amount":     safe_shares,
                        }
                        print(f"  📊 {safe_shares:.4f} shares tracked — take profit at {round(current_odds * (1 + TAKE_PROFIT_PCT))}¢")
                    else:
                        consecutive_losses += 1
                        consecutive_wins   = 0
                        if consecutive_losses >= 3:
                            stop_until = int(time.time()) + 3600
                            print(f"  🛑 3 losses in a row — stopping for 1 hour")
                else:
                    print(f"  → PTB not confirmed — rechecking in {RECHECK_INTERVAL}s")

            if secs_left > BET_SECONDS_BEFORE + 5:
                sleep_for = secs_left - BET_SECONDS_BEFORE - 3
                print(f"  Sleeping {sleep_for:.0f}s...")
                time.sleep(sleep_for)
            else:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopped"); break
        except Exception as e:
            print(f"[!] {e}"); time.sleep(2)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true",
                        help="Download historical data and train model first")
    args = parser.parse_args()

    if args.bootstrap:
        bootstrap()
        print("\nBootstrap done! Now run: python bot.py")
    else:
        run_bot()
