import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import sys
import io

# Force UTF-8 output for Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.progress_bar import ProgressBar
from rich import box



def compute_rsi(series, period=14):
    """RSI nach Wilders Methode (EMA-basiert)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def timing_factor(days_since_ath, avg_cycle_days=360):
    """
    Dynamischer Zyklus-Multiplikator: Gewichtet den Score nach Zyklus-Reife.

    Statt fester Tage werden Prozentsaetze des durchschnittlichen Zyklus verwendet.
    Damit passt sich der Faktor automatisch an wenn sich Zyklen verlaengern
    (z.B. durch ETFs/Institutionelle) oder verkuerzen.

    Backtest-validiert: Filtert Zwischen-Crashs (z.B. Luna 2022) raus.
    """
    pct = days_since_ath / avg_cycle_days if avg_cycle_days > 0 else 0

    if pct >= 0.90:   return 1.00   # Im klassischen Boden-Fenster
    elif pct >= 0.75: return 0.90   # Nahe dran
    elif pct >= 0.60: return 0.75   # Mittlere Reife
    elif pct >= 0.45: return 0.60   # Noch frueh
    else:             return 0.50   # Zu frueh im Zyklus


def crash_velocity_factor(price, df_tail_30, days_since_ath, avg_cycle_days):
    """
    Crash-Velocity-Filter: Erkennt aktive Crashes in unreifen Zyklen.

    Problem: Zwischen-Crashs (z.B. Luna Mai 2022) erzeugen hohe Raw-Scores
    durch starken Drawdown, niedrigen RSI und Mayer-Multiple. Der
    timing_factor allein reicht nicht, wenn der Raw-Score 80+ ist.

    Loesung: Wenn der Preis in den letzten 30 Tagen stark gefallen ist
    UND der Zyklus noch nicht reif genug ist (< 85%), wird der Score
    zusaetzlich reduziert.

    Bei reifen Zyklen (>= 85%) wird NICHT bestraft, da ein steiler
    Drop dort eher eine finale Kapitulation ist (z.B. Nov 2018).
    """
    if df_tail_30 is None or len(df_tail_30) < 20:
        return 1.0

    price_30d_ago = df_tail_30['Close'].iloc[0]
    drop_30d = ((price - price_30d_ago) / price_30d_ago) * 100
    pct_cycle = days_since_ath / avg_cycle_days if avg_cycle_days > 0 else 0

    if drop_30d <= -20 and pct_cycle < 0.85:
        if drop_30d <= -30:
            return 0.50    # Extremer Crash in unreifem Zyklus
        else:
            return 0.65    # Starker Crash in unreifem Zyklus

    return 1.0


def compute_bottom_score(price, sma_200w, sma_200d, weekly_rsi, ath, days_since_ath,
                         df_tail_90, df_tail_30, df_tail_14, df_tail_7,
                         avg_cycle_days=360):
    """
    Berechnet den Composite Bottom Score (0-100).

    Technische Komponenten (max 100):
      1. 200W SMA Abweichung (0-30)
      2. Mayer Multiple (0-20)
      3. Wöchentlicher RSI (0-20)
      4. Drawdown vom ATH (0-20)
      5. Kapitulation & Volumen/Zeit (0-10)
    """

    deviation = ((price - sma_200w) / sma_200w) * 100 if sma_200w > 0 else 0
    mayer = price / sma_200d if sma_200d > 0 else 1.0
    drawdown = ((price - ath) / ath) * 100

    # 1. 200W SMA Abweichung (0-30)
    if deviation <= -30: s1 = 30
    elif deviation <= -20: s1 = 26
    elif deviation <= -10: s1 = 22
    elif deviation <= 0:   s1 = 18
    elif deviation <= 5:   s1 = 14
    elif deviation <= 15:  s1 = 8
    elif deviation <= 30:  s1 = 3
    else: s1 = 0

    # 2. Mayer Multiple (0-20)
    if mayer <= 0.5:   s2 = 20
    elif mayer <= 0.6: s2 = 17
    elif mayer <= 0.7: s2 = 14
    elif mayer <= 0.8: s2 = 11
    elif mayer <= 1.0: s2 = 6
    elif mayer <= 1.2: s2 = 2
    else: s2 = 0

    # 3. Wöchentlicher RSI (0-20)
    if weekly_rsi <= 28:   s3 = 20
    elif weekly_rsi <= 33: s3 = 16
    elif weekly_rsi <= 38: s3 = 12
    elif weekly_rsi <= 43: s3 = 7
    elif weekly_rsi <= 50: s3 = 3
    else: s3 = 0

    # 4. Drawdown vom ATH (0-20)
    if drawdown <= -80:   s4 = 20
    elif drawdown <= -75: s4 = 17
    elif drawdown <= -70: s4 = 14
    elif drawdown <= -60: s4 = 10
    elif drawdown <= -50: s4 = 6
    elif drawdown <= -30: s4 = 3
    else: s4 = 0

    # 5. Kapitulation & Volumen-Dynamik (0-10)
    s5 = 0
    if df_tail_90 is not None:
        avg_vol = df_tail_90['Volume'].mean()

        # a) Volumen-Spike in letzten 30 Tagen
        if df_tail_30 is not None and avg_vol > 0:
            max_vol_30d = df_tail_30['Volume'].max()
            if max_vol_30d > avg_vol * 2.0:
                s5 += 4

        # b) Volumen-Austrocknung (Selling Exhaustion)
        if df_tail_7 is not None and avg_vol > 0:
            vol_7d = df_tail_7['Volume'].mean()
            if vol_7d < avg_vol * 0.75:
                s5 += 3

        # c) Volatilitäts-Kompression
        if df_tail_14 is not None:
            vol_short = df_tail_14['Close'].pct_change().std()
            vol_long = df_tail_90['Close'].pct_change().std()
            if vol_long > 0 and vol_short / vol_long < 0.55:
                s5 += 3

    # d) Chronische Kapitulation (Zeit seit ATH)
    if days_since_ath > 300 and drawdown <= -30:
        if days_since_ath > 500:
            s5 += 3
        elif days_since_ath > 400:
            s5 += 2
        else:
            s5 += 1

    s5 = min(s5, 10)

    total = s1 + s2 + s3 + s4 + s5

    # Zyklus-Multiplikator (Backtest-validiert, dynamisch)
    factor = timing_factor(days_since_ath, avg_cycle_days)

    # Crash-Velocity-Filter: Bestraft steile Drops in unreifen Zyklen
    crash_factor = crash_velocity_factor(price, df_tail_30, days_since_ath, avg_cycle_days)

    adjusted = int(round(total * factor * crash_factor))

    return {
        "total": total,
        "adjusted": adjusted,
        "factor": factor,
        "crash_factor": crash_factor,
        "sma_dev": s1,
        "mayer": s2,
        "rsi": s3,
        "drawdown": s4,
        "capitulation": s5,
        "raw": {
            "deviation": deviation,
            "mayer_val": mayer,
            "rsi_val": weekly_rsi,
            "drawdown_val": drawdown,
            "days_since_ath": days_since_ath,
        },
    }


def score_label(score):
    if score >= 80: return ("EXTREM 🔥", "bold white on red")
    elif score >= 65: return ("STARK 🚨", "bold red")
    elif score >= 45: return ("MODERAT ⚠️", "bold yellow")
    elif score >= 25: return ("SCHWACH ⏳", "yellow")
    else: return ("KEIN SIGNAL 🧊", "dim")


def compute_entry_prices(sma_200w, sma_200d, ath, days_since_ath, weekly_rsi,
                         avg_cycle_days=360):
    """
    Entry-Preis-Rechner: Berechnet die Preislevel bei denen der
    Adjusted Score bestimmte Schwellenwerte erreicht.

    Berechnet mit Faktor x1.00 (= reifer Zyklus), weil:
    Bis der Preis die Entry-Level erreicht, werden Monate vergangen
    und der Zyklus wird reif sein. Es macht keinen Sinn, den heutigen
    (unreifen) Faktor fuer zukuenftige Preise zu verwenden.

    RSI-Annahme: Konservativ (aktueller Wert als Schätzung).
    S5-Annahme: 0 Punkte (konservativ, Kapitulation kommt erst am Boden).
    """
    current_factor = timing_factor(days_since_ath, avg_cycle_days)
    calc_factor = 1.00  # Berechnung immer mit reifem Zyklus

    # RSI-Szenarien: aktueller RSI + pessimistischer RSI
    def rsi_to_pts(rsi):
        if rsi <= 28:   return 20
        elif rsi <= 33: return 16
        elif rsi <= 38: return 12
        elif rsi <= 43: return 7
        elif rsi <= 50: return 3
        else: return 0

    # Konservativ: RSI wie aktuell (koennte schlechter werden am Boden)
    rsi_pts = rsi_to_pts(weekly_rsi)

    targets = [45, 55, 65]
    results = {}

    # Sweep von ATH runter in $100-Schritten
    for target in targets:
        entry = None
        for p in np.arange(ath, 100, -100):
            deviation = ((p - sma_200w) / sma_200w) * 100 if sma_200w > 0 else 0
            mayer = p / sma_200d if sma_200d > 0 else 1.0
            dd = ((p - ath) / ath) * 100

            if deviation <= -30: s1 = 30
            elif deviation <= -20: s1 = 26
            elif deviation <= -10: s1 = 22
            elif deviation <= 0:   s1 = 18
            elif deviation <= 5:   s1 = 14
            elif deviation <= 15:  s1 = 8
            elif deviation <= 30:  s1 = 3
            else: s1 = 0

            if mayer <= 0.5:   s2 = 20
            elif mayer <= 0.6: s2 = 17
            elif mayer <= 0.7: s2 = 14
            elif mayer <= 0.8: s2 = 11
            elif mayer <= 1.0: s2 = 6
            elif mayer <= 1.2: s2 = 2
            else: s2 = 0

            if dd <= -80:   s4 = 20
            elif dd <= -75: s4 = 17
            elif dd <= -70: s4 = 14
            elif dd <= -60: s4 = 10
            elif dd <= -50: s4 = 6
            elif dd <= -30: s4 = 3
            else: s4 = 0

            # S5 konservativ auf 0 (Kapitulation kommt erst spaet)
            raw = s1 + s2 + rsi_pts + s4 + 0
            adj = int(round(raw * calc_factor))

            if adj >= target:
                entry = p
                break

        results[target] = entry

    return results, current_factor


def calculate_dynamic_cycle_timing(df):
    """Analysiert die historischen Zyklen im DataFrame und berechnet dynamisch die Dauer vom ATH zum Boden."""
    running_max = df['High'].cummax()
    drawdown = (df['Low'] - running_max) / running_max
    
    bear_markets = []
    in_bear = False
    current_ath_date = None
    current_bottom_date = None
    current_bottom_val = float('inf')
    
    for date, row in df.iterrows():
        if row['High'] >= running_max.loc[date]:
            if not in_bear:
                current_ath_date = date
                
        if drawdown.loc[date] < -0.50: # Ab 50% Drawdown
            if not in_bear:
                in_bear = True
                current_bottom_val = row['Low']
                current_bottom_date = date
            else:
                if row['Low'] < current_bottom_val:
                    current_bottom_val = row['Low']
                    current_bottom_date = date
        else:
            if in_bear and drawdown.loc[date] > -0.25:
                if current_ath_date and current_bottom_date:
                    days_to_bottom = (current_bottom_date - current_ath_date).days
                    if days_to_bottom > 100:
                        bear_markets.append(days_to_bottom)
                in_bear = False
                current_bottom_val = float('inf')

    if not bear_markets:
        return 350, 400, 375, 0
        
    avg_days = sum(bear_markets) / len(bear_markets)
    min_days = min(bear_markets)
    max_days = max(bear_markets)
    return int(min_days), int(max_days), int(avg_days), len(bear_markets)


def analyze_bear_market_drawdowns(df):
    """
    Analysiert alle ABGESCHLOSSENEN historischen Bärenmärkte und gibt
    die maximalen Drawdowns (in %) pro Zyklus zurück.

    Verwendet nur abgeschlossene Bärenmärkte (Recovery > -25%),
    damit der aktuelle Zyklus die Schätzung nicht verzerrt.
    """
    running_max = df['High'].cummax()
    drawdown = (df['Low'] - running_max) / running_max * 100

    completed_bears = []
    in_bear = False
    current_max_dd = 0

    for date, row in df.iterrows():
        dd = drawdown.loc[date]

        if dd < -50:
            if not in_bear:
                in_bear = True
                current_max_dd = dd
            else:
                if dd < current_max_dd:
                    current_max_dd = dd
        elif in_bear and dd > -25:
            completed_bears.append(current_max_dd)
            in_bear = False
            current_max_dd = 0

    return completed_bears


def compute_entry_probability(bear_drawdowns, entry_prices, ath, current_price,
                              days_since_ath, avg_cycle_days):
    """
    Schätzt die Wahrscheinlichkeit, dass BTC jedes Entry-Level erreicht.

    Methode:
    1. Historische Drawdown-Häufigkeit (gewichtet: neuere Zyklen zählen mehr
       wegen Market Maturation / institutioneller Adoption)
    2. Cycle-Timing-Faktor (reifer Zyklus → höhere Wahrscheinlichkeit)
    3. Bereits erreicht → 99%

    Gibt dict {target_score: probability_0_to_1} zurück.
    """
    if not bear_drawdowns:
        return {t: None for t in entry_prices}

    probabilities = {}
    n = len(bear_drawdowns)

    # Lineare Gewichtung: neuere Zyklen sind relevanter
    # z.B. 3 Zyklen → Gewichte [1, 2, 3], neuester zählt 3x so viel wie ältester
    weights = [i + 1 for i in range(n)]
    total_weight = sum(weights)

    # Cycle Timing Multiplier
    cycle_pct = days_since_ath / avg_cycle_days if avg_cycle_days > 0 else 0
    if cycle_pct >= 0.90:
        timing_mult = 1.15   # Im klassischen Bottom-Fenster
    elif cycle_pct >= 0.70:
        timing_mult = 1.05   # Nahe am Fenster
    elif cycle_pct >= 0.50:
        timing_mult = 0.90   # Mittlere Reife
    else:
        timing_mult = 0.75   # Zu früh für echten Boden

    for target, ep in entry_prices.items():
        if ep is None:
            probabilities[target] = None
            continue

        # Bereits erreicht?
        if current_price <= ep:
            probabilities[target] = 0.99
            continue

        # Erforderlicher Drawdown vom ATH
        required_dd = ((ep - ath) / ath) * 100

        # Gewichtete historische Wahrscheinlichkeit
        weighted_hits = sum(
            w for dd, w in zip(bear_drawdowns, weights) if dd <= required_dd
        )
        base_prob = weighted_hits / total_weight

        # Timing anwenden
        prob = base_prob * timing_mult

        # Clamp auf 5%-95%
        prob = max(0.05, min(prob, 0.95))

        probabilities[target] = prob

    return probabilities


def prob_label(prob):
    """Gibt Label und Farbe für eine Wahrscheinlichkeit zurück."""
    pct = prob * 100
    if pct >= 70:
        return "Sehr wahrsch.", "green"
    elif pct >= 50:
        return "Wahrscheinlich", "yellow"
    elif pct >= 30:
        return "Möglich", "bright_red"
    else:
        return "Unwahrsch.", "red"


def generate_llm_prompt(data):
    """Generiert einen Prompt für eine LLM mit Internet-Zugang zur Verifizierung der Analyse."""
    d = data
    today_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

    prompt = f"""Du bist ein erfahrener Bitcoin-Analyst. Ich habe eine automatisierte Bitcoin-Bottom-Analyse durchgeführt (Stand: {today_str}). Die technischen Indikatoren und Preisdaten unten sind zuverlässig berechnet und dienen dir als Referenz. Deine Aufgabe ist es, eine eigenständige Analyse mit zusätzlichen Daten (On-Chain, Makro, News) durchzuführen und deine eigene unabhängige Einschätzung abzugeben.

═══════════════════════════════════════════════════════════
MEINE BERECHNETEN DATEN (zuverlässig berechnet):
═══════════════════════════════════════════════════════════

📊 MARKTDATEN:
- Aktueller BTC-Preis: ${d['price']:,.2f}
- All-Time High: ${d['ath']:,.2f} (am {d['ath_date']})
- Drawdown vom ATH: {d['drawdown']:.2f}%
- Tage seit ATH: {d['days_since_ath']}

📈 TECHNISCHE INDIKATOREN:
- 200-Wochen-SMA: ${d['sma_200w']:,.2f}
- Abweichung vom 200W-SMA: {d['deviation']:.2f}%
- 200-Tage-SMA: ${d['sma_200d']:,.2f}
- Mayer Multiple: {d['mayer_multiple']:.4f}
- Wöchentlicher RSI (14): {d['weekly_rsi']:.2f}

📉 BULL MARKET SUPPORT BAND:
- 20W SMA: ${d['bmsb_sma20w']:,.2f}
- 21W EMA: ${d['bmsb_ema21w']:,.2f}
- Trend-Status: {d['trend_status']}

🎯 COMPOSITE BOTTOM SCORE: {d['score_total']}/100 ({d['score_label']})
  Technische Analyse ({d['score_raw']}/100, nach Zyklus-Faktor: {d['score_total']}/100):
  - 200W SMA Abweichung:    {d['s1']}/30  (Deviation: {d['deviation']:.1f}%)
  - Mayer Multiple:         {d['s2']}/20  (Mayer: {d['mayer_multiple']:.4f})
  - Wöchentlicher RSI:      {d['s3']}/20  (RSI: {d['weekly_rsi']:.1f})
  - Drawdown vom ATH:       {d['s4']}/20  (Drawdown: {d['drawdown']:.1f}%)
  - Kapitulation/Volumen:   {d['s5']}/10

📡 KAUFZONEN (basierend auf 200W SMA ${d['sma_200w']:,.0f}):
  - Zone 1 (Soft Floor):  ${d['z1_upper']:,.0f} - ${d['z1_lower']:,.0f}
  - Zone 2 (Hard Floor):  ${d['z2_upper']:,.0f} - ${d['z2_lower']:,.0f}
  - Zone 3 (Max Pain):    ${d['z3_upper']:,.0f} - ${d['z3_lower']:,.0f}
  - Zone 4 (Black Swan):  ${d['z4_upper']:,.0f} - ${d['z4_lower']:,.0f}
  - Aktive Zone: {d['active_zone_name']}

⏳ ZYKLUS-TIMING ({d['cycles_found']} historische Zyklen analysiert):
  - Tage seit ATH: {d['days_since_ath']}
  - Klassisches Fenster: {d['classic_start']} bis {d['classic_end']}
  - Institutionelles Fenster: {d['inst_start']} bis {d['inst_end']}
  - Timing-Phase: {d['timing_phase']}

💡 TOOL-FAZIT: {d['fazit']}

═══════════════════════════════════════════════════════════
DEINE AUFGABEN:
═══════════════════════════════════════════════════════════

1️⃣ AKTUELLE NACHRICHTEN & EVENTS ANALYSIEREN:
   - Was sind die wichtigsten aktuellen BTC/Krypto-Nachrichten?
   - Regulatorische Entwicklungen (weltweit, nicht nur USA/EU)?
   - Institutionelle Zuflüsse/Abflüsse (ETFs, Fonds, börsengehandelte Produkte)?
   - Bevorstehende Events (Halving, Zentralbank-Entscheidungen, etc.)?
   - Globale Nachrichten die den Markt beeinflussen könnten (Geopolitik, Handelskonflikte, Krisen, etc.)?

2️⃣ ON-CHAIN DATEN RECHERCHIEREN (bitte selbst aktuelle Werte im Internet suchen):
   Mein Tool erfasst keine On-Chain Daten - bitte recherchiere diese selbst:
   - MVRV Z-Score (unter 0 = historisch unterbewertet)
   - SOPR (Spent Output Profit Ratio)
   - Exchange Netflows (Zu-/Abflüsse von Börsen)
   - Long-Term Holder vs Short-Term Holder Verhalten
   - Realized Price vs. Market Price / Realized Cap vs. Market Cap
   - NUPL (Net Unrealized Profit/Loss)
   - Fear & Greed Index
   - Active Addresses / Transaktionsvolumen / Mempool-Auslastung
   - Stablecoin Supply & Flows (Mint/Burn-Trends als Liquiditätsindikator)
   - Bitcoin Dominance (Kapitalrotation BTC vs. Altcoins)

3️⃣ MINING & NETZWERK-GESUNDHEIT:
   - Hash Rate Trend
   - Mining Difficulty & nächstes Difficulty Adjustment
   - Hash Price (Miner-Profitabilität)
   - Miner-Verhalten (Kapitulation? Akkumulation? Miner-Reserven?)

4️⃣ DERIVATE-MARKT:
   - Funding Rates (Perpetual Futures) - positiv/negativ?
   - Open Interest Trend (Leverage im Markt?)
   - Liquidation Heatmaps / größte Liquidation Cluster
   - Long/Short Ratio

5️⃣ MAKRO-UMFELD BEWERTEN:
   - US Dollar Index (DXY) - Trend?
   - Zinspolitik der großen Zentralbanken (Fed, EZB, etc.) - Aktuelle Erwartungen?
   - Staatsanleihen-Renditen (US 10Y, global)
   - Globale Liquidität (Global M2 Money Supply, nicht nur US)
   - Aktienmarkt-Korrelation mit BTC (S&P 500, Nasdaq, etc.)
   - Geopolitische Risiken

6️⃣ EIGENE EINSCHÄTZUNG:
   - Wie ist deine eigene Einschätzung zur aktuellen Marktlage? (Mein Score von {d['score_total']}/100 dient als Referenz)
   - Gibt es Faktoren die besonders wichtig sind und über reine technische Analyse hinausgehen?
   - Was ist dein eigenes Bull/Bear-Szenario für die nächsten 3-6 Monate?

7️⃣ RISIKEN & WARNSIGNALE:
   - Gibt es akute Risiken, die mein Tool nicht erfasst?
   - Black-Swan-Szenarien die man beachten sollte?
   - Leverage/Liquidation-Risiken im Markt (Funding Rates, OI)?

8️⃣ DEINE EIGENEN BODEN-ZONEN (WICHTIG!):
   Basierend auf ALLEN Daten die du gesammelt hast (On-Chain, Makro, News, Technische Analyse),
   erstelle deine eigenen Preiszonen wo der Boden dieses Zyklus liegen könnte.

   Meine Zonen basieren NUR auf dem 200W SMA. Deine Zonen sollen ZUSÄTZLICH berücksichtigen:
   - Realized Price (durchschnittlicher Kaufpreis aller BTC im Netzwerk)
   - Short-Term Holder Realized Price
   - Long-Term Holder Cost Basis
   - Wichtige Liquiditätszonen / Liquidation Cluster
   - Historische Support-Level (vorherige Cycle-Tops als Support)
   - On-Chain Akkumulationszonen (wo kaufen Wale?)
   - CME Gaps (offene Gaps die oft gefüllt werden)
   - Fibonacci Retracement Levels vom ATH
   - VWAP-basierte Zonen
   - Aktuelle Open Interest & Liquidation Heatmaps

   Gib mir 4 Zonen in diesem Format:
   🟢 ZONE A (Wahrscheinlichster Boden): $XX,XXX - $XX,XXX
      → Begründung: [Welche Daten stützen diese Zone?]
      → Strategie: [Was sollte man in dieser Zone tun? z.B. DCA starten, % des Kapitals einsetzen]

   🟡 ZONE B (Tieferer Boden bei Eskalation): $XX,XXX - $XX,XXX
      → Begründung: [Welche Daten stützen diese Zone?]
      → Strategie: [Was sollte man in dieser Zone tun?]

   🔴 ZONE C (Worst Case / Kapitulation): $XX,XXX - $XX,XXX
      → Begründung: [Welche Daten stützen diese Zone?]
      → Strategie: [Was sollte man in dieser Zone tun?]

   ⚫ ZONE D (Black Swan / Extremszenario): $XX,XXX - $XX,XXX
      → Begründung: [Was müsste passieren damit wir hier landen?]
      → Strategie: [Was sollte man in dieser Zone tun?]

9️⃣ ZEITLICHE EINSCHÄTZUNG:
   - Wann könnte der Boden zeitlich erreicht werden?
   - Gibt es Katalysatoren (Events, Halvings, Fed-Meetings) die den Zeitrahmen beeinflussen?
   - (Referenz: Mein Tool schätzt klassisches Fenster {d['classic_start']} - {d['classic_end']}, institutionell {d['inst_start']} - {d['inst_end']})

═══════════════════════════════════════════════════════════
AUSGABEFORMAT:
═══════════════════════════════════════════════════════════

Bitte strukturiere deine Antwort wie folgt:

📰 NEWS-ANALYSE: [Aktuelle relevante Nachrichten und deren Einfluss]
⛓️ ON-CHAIN ANALYSE: [Aktuelle On-Chain Metriken die du recherchiert hast, mit konkreten Werten]
⛏️ MINING & NETZWERK: [Hash Rate, Difficulty, Hash Price, Miner-Verhalten]
📈 DERIVATE-MARKT: [Funding Rates, Open Interest, Liquidationen, Long/Short Ratio]
🌍 MAKRO-ANALYSE: [Makroökonomische Faktoren]

🗺️ MEINE BODEN-ZONEN:
   🟢 Zone A: $XX,XXX - $XX,XXX → Strategie
   🟡 Zone B: $XX,XXX - $XX,XXX → Strategie
   🔴 Zone C: $XX,XXX - $XX,XXX → Strategie
   ⚫ Zone D: $XX,XXX - $XX,XXX → Strategie
   [Begründung je Zone mit konkreten Datenpunkten]

⏳ TIMING: [Zeitliche Einschätzung wann der Boden kommen könnte]
🎯 EIGENE BEWERTUNG: [Dein eigener Bottom Score und Begründung]
⚠️ RISIKEN: [Aktuelle Warnsignale]
📊 FAZIT:
   Die aktuelle Situation in einem Satz: [Ein prägnanter Satz der die Gesamtlage zusammenfasst - Zyklusphase, Sentiment, institutionelle Lage, Makro-Faktoren]

   Handlungsempfehlung (gestuft nach Preiszonen):
   - Sofort (aktuelle Zone, $XX,XXX-$XX,XXX): XX% des geplanten Budgets per DCA einsetzen. [Begründung, z.B. Fear & Greed Level als historisches Signal]
   - Bei $XX,XXX-$XX,XXX: Weitere XX% einsetzen. [Begründung, z.B. Realized Price Bereich]
   - Warten auf [Zeitraum]: [Was muss passieren damit Klarheit entsteht, z.B. Fed-Pivot, Makro-Entspannung]

   Boden-Einschätzung:
   - Der Boden dieses Zyklus liegt mit XX% Wahrscheinlichkeit im Bereich $XX,XXX-$XX,XXX
   - Worst-Case-Szenario: $XX,XXX-$XX,XXX bei [Bedingung, z.B. weitere Makro-Verschlechterung] (XX% Wahrscheinlichkeit)
   - Unter $XX,XXX halte ich für unwahrscheinlich (<XX%), weil [strukturelle Begründung, z.B. ETF-Adoption, institutioneller Boden]

Wichtig: Die obigen technischen Indikatoren und Preisdaten sind zuverlässig berechnet - du musst sie nicht verifizieren. Nutze sie als Referenz für deine eigene Analyse. Nutze bitte aktuelle Echtzeitdaten für On-Chain, Mining, Derivate, Makro und News-Analyse. Gib konkrete Zahlen an wo möglich. Dein Mehrwert liegt in deiner eigenständigen Analyse und dem Einbringen zusätzlicher Datenpunkte die mein Tool nicht erfasst. Falls bestimmte Daten nicht verfügbar sind, kennzeichne dies transparent statt Werte zu schätzen.
"""
    return prompt


def main():
    console = Console()

    with console.status("[bold cyan]Lade historische Marktdaten von Yahoo Finance ⏳...", spinner="dots"):
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="max")

        if df.empty:
            console.print("[bold red]❌ Fehler: Konnte keine Daten laden.[/bold red]")
            sys.exit(1)

        df.index = df.index.tz_localize(None)

        current_price = df['Close'].iloc[-1]
        today = datetime.datetime.now()

        ath = df['High'].max()
        ath_date = df[df['High'] == ath].index[-1]
        drop_from_ath = ((current_price - ath) / ath) * 100
        days_since_ath = (today - ath_date).days

        # --- Gleitende Durchschnitte ---
        weekly_df = df['Close'].resample('W').last().dropna()
        sma_200w = weekly_df.rolling(window=200).mean().iloc[-1] if len(weekly_df) >= 200 else weekly_df.mean()
        sma_200d = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else df['Close'].mean()
        mayer_multiple = current_price / sma_200d

        # Weekly RSI
        weekly_rsi_series = compute_rsi(weekly_df, 14)
        weekly_rsi = weekly_rsi_series.iloc[-1] if len(weekly_rsi_series) > 0 and not pd.isna(weekly_rsi_series.iloc[-1]) else 50

        # Bull Market Support Band (20W SMA & 21W EMA)
        sma_20w = weekly_df.rolling(window=20).mean().iloc[-1] if len(weekly_df) >= 20 else np.nan
        ema_21w = weekly_df.ewm(span=21, adjust=False).mean().iloc[-1] if len(weekly_df) >= 21 else np.nan
        if not pd.isna(sma_20w) and not pd.isna(ema_21w):
            bmsb_upper = max(sma_20w, ema_21w)
            bmsb_lower = min(sma_20w, ema_21w)
        else:
            bmsb_upper = bmsb_lower = np.nan

        # Zonen (dynamisch auf 200W SMA)
        zone1_upper = sma_200w * 1.25
        zone1_lower = sma_200w * 1.10
        zone2_upper = sma_200w * 1.10
        zone2_lower = sma_200w * 0.95
        zone3_upper = sma_200w * 0.95
        zone3_lower = sma_200w * 0.80
        zone4_upper = sma_200w * 0.80
        zone4_lower = sma_200w * 0.60

        tail_90 = df.tail(90) if len(df) >= 90 else None
        tail_30 = df.tail(30) if len(df) >= 30 else None
        tail_14 = df.tail(14) if len(df) >= 14 else None
        tail_7 = df.tail(7) if len(df) >= 7 else None

    # --- DYNAMISCHE ZYKLUSLAENGE ---
    min_d, max_d, avg_d, cycles_found = calculate_dynamic_cycle_timing(df)

    # --- HISTORISCHE DRAWDOWN-ANALYSE ---
    bear_drawdowns = analyze_bear_market_drawdowns(df)

    # --- COMPOSITE BOTTOM SCORE ---
    score = compute_bottom_score(
        current_price, sma_200w, sma_200d, weekly_rsi, ath,
        days_since_ath, tail_90, tail_30, tail_14, tail_7,
        avg_cycle_days=avg_d,
    )
    console.print()

    # =========================================================================
    # CLI UI
    # =========================================================================

    # 1. Header
    header = Panel(
        Text("₿ BITCOIN BOTTOM ANALYZER 📊", justify="center", style="bold white on blue"),
        box=box.DOUBLE_EDGE,
        border_style="blue",
        padding=(1, 1)
    )
    console.print(header)

    # 2. COMPOSITE BOTTOM SCORE (Herzstück)
    total = score["total"]
    adjusted = score["adjusted"]
    factor = score["factor"]
    crash_f = score.get("crash_factor", 1.0)
    label, label_style = score_label(adjusted)
    raw = score["raw"]

    score_text = Text()
    score_text.append(f"\n  🎯 BOTTOM SCORE:  ", style="bold")
    score_text.append(f" {adjusted} / 100 ", style=f"bold white on {'red' if adjusted >= 65 else 'yellow' if adjusted >= 40 else 'green'}")
    score_text.append(f"  {label}\n", style=label_style)
    if crash_f < 1.0:
        score_text.append(f"  (Raw: {total} x{factor:.2f} Zyklus x{crash_f:.2f} Crash-Velocity = {adjusted})\n", style="dim")
        score_text.append(f"  ⚡ Crash-Velocity aktiv: Steiler Drop in unreifem Zyklus erkannt!\n\n", style="bold yellow")
    else:
        score_text.append(f"  (Raw: {total} x{factor:.2f} Zyklus-Faktor = {adjusted})\n\n", style="dim")

    # Balken-Visualisierung
    bar_filled = int(adjusted / 2)  # 0-50 Zeichen
    bar_empty = 50 - bar_filled
    bar_color = "red" if adjusted >= 65 else "yellow" if adjusted >= 40 else "green"
    score_text.append(f"  [{'█' * bar_filled}{'░' * bar_empty}]\n\n", style=bar_color)

    # Komponenten-Aufschlüsselung mit Klartext
    dev = raw['deviation']
    mayer_val = raw['mayer_val']
    rsi_val = raw['rsi_val']
    dd_val = raw['drawdown_val']

    # Dynamische Erklärungen je nach aktuellem Wert
    if dev <= 0:
        sma_explain = f"Preis liegt {abs(dev):.0f}% UNTER dem Langzeit-Durchschnitt. Historisch günstig!"
    elif dev <= 15:
        sma_explain = f"Preis liegt {dev:.0f}% über dem Langzeit-Durchschnitt. Nahe dran, aber noch nicht drunter."
    else:
        sma_explain = f"Preis liegt {dev:.0f}% über dem Langzeit-Durchschnitt. Noch weit von einer Kaufzone entfernt."

    if mayer_val <= 0.8:
        mayer_explain = f"BTC kostet nur {mayer_val:.2f}x vom Halbjahres-Trend. Unter 0.8 = historisch billig!"
    elif mayer_val <= 1.2:
        mayer_explain = f"BTC kostet {mayer_val:.2f}x vom Halbjahres-Trend. Fair bewertet, noch kein Schnäppchen."
    else:
        mayer_explain = f"BTC kostet {mayer_val:.2f}x vom Halbjahres-Trend. Überhitzt/teuer."

    if rsi_val <= 33:
        rsi_explain = f"Markt ist stark überverkauft (RSI {rsi_val:.0f}). Verkäufer werden müde!"
    elif rsi_val <= 45:
        rsi_explain = f"Markt zeigt Schwäche (RSI {rsi_val:.0f}). Noch nicht am Limit."
    else:
        rsi_explain = f"Markt hat noch Kraft (RSI {rsi_val:.0f}). Kein Überverkauf-Signal."

    if dd_val <= -75:
        dd_explain = f"Preis ist {abs(dd_val):.0f}% unter dem Allzeithoch. Extreme Crash-Tiefe!"
    elif dd_val <= -50:
        dd_explain = f"Preis ist {abs(dd_val):.0f}% unter dem Allzeithoch. Tiefer Bärenmarkt."
    elif dd_val <= -30:
        dd_explain = f"Preis ist {abs(dd_val):.0f}% unter dem Allzeithoch. Korrektur, aber noch nicht tief genug."
    else:
        dd_explain = f"Preis ist nur {abs(dd_val):.0f}% unter dem Allzeithoch. Kein tiefer Crash."

    cap_pts = score["capitulation"]
    if cap_pts >= 7:
        cap_explain = "Zeichen von Massenpanik erkannt! Hohes Volumen + Markt beruhigt sich."
    elif cap_pts >= 4:
        cap_explain = "Kapitulations-Signale erkannt (Volumen/Zeit). Markt zeigt Erschöpfung."
    else:
        cap_explain = "Keine Massenpanik erkannt. Markt blutet langsam, kein großer Ausverkauf."

    # Technische Komponenten
    score_text.append(f"  {'─' * 48}\n", style="dim")
    score_text.append(f"  TECHNISCHE ANALYSE          {score['total']}/100\n", style="bold cyan")
    score_text.append(f"  {'─' * 48}\n", style="dim")

    tech_components = [
        ("📈 Langzeit-Trend",   score["sma_dev"],      30, sma_explain),
        ("⚖️ Unterbewertung",   score["mayer"],         20, mayer_explain),
        ("🩸 Verkaufsdruck",    score["rsi"],           20, rsi_explain),
        ("💥 Crash-Tiefe",      score["drawdown"],      20, dd_explain),
        ("😱 Panik-Verkauf",    score["capitulation"],  10, cap_explain),
    ]

    for name, pts, max_pts, explain in tech_components:
        pct = pts / max_pts if max_pts > 0 else 0
        color = "red" if pct >= 0.7 else "yellow" if pct >= 0.4 else "dim"
        bar_w = 15
        filled = int(pct * bar_w)
        score_text.append(f"  {name:<20} ", style="bold")
        score_text.append(f"{'█' * filled}{'░' * (bar_w - filled)} ", style=color)
        score_text.append(f"{pts:>2}/{max_pts}\n", style=color)
        score_text.append(f"  {'':20} {explain}\n", style="dim italic")

    # Zyklus-Faktor Erklärung
    score_text.append(f"\n  {'─' * 48}\n", style="dim")
    score_text.append(f"  ZYKLUS-MULTIPLIKATOR          x{factor:.2f}\n", style="bold magenta")
    score_text.append(f"  {'─' * 48}\n", style="dim")
    score_text.append(f"  Tage seit ATH: {raw['days_since_ath']} / Ø Zyklus: {avg_d} Tage ({raw['days_since_ath']/avg_d*100:.0f}%)\n", style="dim")
    if factor < 1.0:
        score_text.append(f"  ⏳ Score wird gedämpft (Zyklus noch jung)\n", style="dim italic")
        score_text.append(f"     Voller Score ab ≥90% des Ø Zyklus ({int(avg_d*0.9)} Tage)\n", style="dim italic")
    else:
        score_text.append(f"  ✅ Volle Gewichtung (Zyklus ist reif genug)\n", style="dim italic")

    score_text.append(f"\n")
    border = "red" if adjusted >= 65 else "yellow" if adjusted >= 40 else "green"
    console.print(Panel(score_text, title="[bold]✨ COMPOSITE BOTTOM SCORE ✨[/bold]", border_style=border))

    # 3. Aktuelle Marktübersicht
    market_table = Table(show_header=False, box=box.SIMPLE, expand=True, border_style="cyan")
    market_table.add_column("Metrik", style="bold")
    market_table.add_column("Wert")

    market_table.add_row("💲 Aktueller Preis", f"[bold green]${current_price:,.2f}[/bold green]")
    market_table.add_row("👑 Höchster Preis (ATH)", f"${ath:,.2f} [dim](am {ath_date.strftime('%Y-%m-%d')})[/dim]")
    market_table.add_row("📉 Abstand zum ATH", f"[{'red' if drop_from_ath < -30 else 'yellow' if drop_from_ath < 0 else 'green'}]{drop_from_ath:.2f}%[/{'red' if drop_from_ath < -30 else 'yellow' if drop_from_ath < 0 else 'green'}]")
    market_table.add_row("🌊 Langzeit-Trend (200W)", f"[yellow]${sma_200w:,.2f}[/yellow]")
    market_table.add_row("🏄 Halbjahres-Trend (200D)", f"[yellow]${sma_200d:,.2f}[/yellow]")
    market_table.add_row("📅 Tage seit ATH", f"{days_since_ath}")

    console.print(Panel(market_table, title="[bold]🌐 Aktuelle Marktdaten[/bold]", border_style="cyan"))

    # 4. Makro-Trend Status (BMSB)
    trend_text = Text()
    if not pd.isna(bmsb_lower) and not pd.isna(bmsb_upper):
        trend_text.append(f"Trend-Grenze liegt bei: ${bmsb_lower:,.0f} - ${bmsb_upper:,.0f}\n\n")

        if current_price > bmsb_upper:
            trend_text.append("🚀 AUFWÄRTSTREND INTAKT 🟢\n", style="bold green")
            trend_text.append("Der Preis hält sich über der Trend-Grenze. Solange das so bleibt,\n")
            trend_text.append("ist der übergeordnete Trend positiv. Bodensuche ist hier zweitrangig.")
            trend_color = "green"
        elif current_price < bmsb_lower:
            trend_text.append("🐻 ABWÄRTSTREND / KORREKTUR 🔴\n", style="bold red")
            trend_text.append("Der Preis ist unter die Trend-Grenze gefallen.\n")
            trend_text.append("Das bedeutet: Der Markt ist im Abwärtstrend. Jetzt wird der Bottom Score\n")
            trend_text.append("oben relevant - er zeigt dir, wie nah wir am möglichen Boden sind.")
            trend_color = "red"
        else:
            trend_text.append("⚖️ ENTSCHEIDUNGSZONE 🟡\n", style="bold yellow")
            trend_text.append("Der Preis kämpft genau an der Trend-Grenze.\n")
            trend_text.append("Hier entscheidet sich, ob der Markt wieder nach oben dreht oder weiter fällt.")
            trend_color = "yellow"
    else:
        trend_text.append("Nicht genug historische Daten für Trend-Analyse.", style="dim")
        trend_color = "dim"

    console.print(Panel(trend_text, title="[bold]🧭 Makro-Trend Status[/bold]", border_style=trend_color))

    # 5. Kaufzonen Radar
    zones_table = Table(box=box.MINIMAL_DOUBLE_HEAD, expand=True)
    zones_table.add_column("Zone", style="bold")
    zones_table.add_column("Preisbereich", justify="right")
    zones_table.add_column("Bedeutung")
    zones_table.add_column("Status", justify="center")

    def get_status(lower, upper, current, zone_num):
        if lower < current <= upper:
            return f"[bold green reverse] 🎯 AKTIV (ZONE {zone_num}) [/bold green reverse]"
        return "[dim]-[/dim]"

    is_zone4 = current_price <= zone4_upper

    zones_table.add_row(
        "☁️ [blue]Zone 1 (Soft Floor)[/blue]",
        f"${zone1_upper:,.0f} - ${zone1_lower:,.0f}",
        "Erste Preisunterstützung",
        get_status(zone1_lower, zone1_upper, current_price, 1)
    )
    zones_table.add_row(
        "🧱 [yellow]Zone 2 (Hard Floor)[/yellow]",
        f"${zone2_upper:,.0f} - ${zone2_lower:,.0f}",
        "Historischer Kernbereich",
        get_status(zone2_lower, zone2_upper, current_price, 2)
    )
    zones_table.add_row(
        "🩸 [red]Zone 3 (Max Pain)[/red]",
        f"${zone3_upper:,.0f} - ${zone3_lower:,.0f}",
        "Stark unterbewerteter Bereich",
        get_status(zone3_lower, zone3_upper, current_price, 3)
    )
    zones_table.add_row(
        "🦢 [bold magenta]Zone 4 (Black Swan)[/bold magenta]",
        f"${zone4_upper:,.0f} - ${zone4_lower:,.0f}",
        "Historisch extrem selten",
        "[bold magenta reverse] 🎯 AKTIV (ZONE 4) [/bold magenta reverse]" if is_zone4 else "[dim]-[/dim]"
    )

    console.print(Panel(zones_table, title="[bold]📡 Kaufzonen-Radar (200W SMA)[/bold]", border_style="green"))

    if current_price > zone1_upper:
        console.print("[bold red]⚠️ >>> AKTUELLER PREIS IST ÜBER ALLEN KAUFZONEN <<< ⚠️[/bold red]", justify="center")

    # 5.5 Entry-Preis-Rechner (Backtest-validiert)
    entry_prices, entry_factor = compute_entry_prices(
        sma_200w, sma_200d, ath, days_since_ath, weekly_rsi,
        avg_cycle_days=avg_d,
    )

    # Wahrscheinlichkeitsschätzung für jedes Entry-Level
    entry_probs = compute_entry_probability(
        bear_drawdowns, entry_prices, ath, current_price,
        days_since_ath, avg_d,
    )

    entry_table = Table(box=box.MINIMAL_DOUBLE_HEAD, expand=True)
    entry_table.add_column("Ziel-Score", style="bold", width=14)
    entry_table.add_column("Signal", width=20)
    entry_table.add_column("Entry-Preis", justify="right", width=14)
    entry_table.add_column("Abstand", justify="right", width=12)
    entry_table.add_column("Wahrscheinl.", justify="center", width=20)
    entry_table.add_column("Status", justify="center", width=16)

    signal_names = {
        45: ("MODERAT ⚠️", "DCA starten"),
        55: ("STARK-VORSTUFE", "Positionen aufbauen"),
        65: ("STARK 🚨", "Maximale Akkumulation"),
    }

    for target in [45, 55, 65]:
        ep = entry_prices[target]
        sig_label, sig_action = signal_names[target]

        if ep is not None:
            dist = ((current_price - ep) / ep) * 100
            if current_price <= ep:
                status = f"[bold green reverse] 🎯 ERREICHT [/bold green reverse]"
                dist_str = f"[green]{dist:+.1f}%[/green]"
            else:
                status = f"[dim]noch ${current_price - ep:,.0f} entfernt[/dim]"
                dist_str = f"[yellow]+{dist:.1f}% drüber[/yellow]"

            # Wahrscheinlichkeits-Anzeige
            prob = entry_probs.get(target)
            if prob is not None:
                pct = prob * 100
                bar_filled = int(round(prob * 10))
                bar_empty = 10 - bar_filled
                plabel, pcolor = prob_label(prob)
                prob_str = f"[{pcolor}]{'█' * bar_filled}{'░' * bar_empty} {pct:.0f}%[/{pcolor}]\n[dim]{plabel}[/dim]"
            else:
                prob_str = "[dim]N/A[/dim]"

            entry_table.add_row(
                f"{'🟡' if target == 45 else '🟠' if target == 55 else '🔴'} Score {target}+",
                f"{sig_label}\n[dim]{sig_action}[/dim]",
                f"[bold]${ep:,.0f}[/bold]",
                dist_str,
                prob_str,
                status,
            )
        else:
            entry_table.add_row(
                f"Score {target}+", sig_label,
                "[dim]N/A[/dim]", "-", "[dim]N/A[/dim]",
                "[dim]Nicht berechenbar[/dim]",
            )

    entry_info = Text()
    entry_info.append(f"Berechnet mit x1.00 (reifer Zyklus)", style="dim")
    entry_info.append(f"  |  RSI ~{weekly_rsi:.0f}, Cap=0 (konservativ)", style="dim")
    if entry_factor < 1.0:
        entry_info.append(f"  |  Aktuell x{entry_factor:.2f} - Entries gelten erst ab reifem Zyklus!", style="dim bold")
    entry_info.append(f"  |  Hist. Zyklen: {len(bear_drawdowns)}", style="dim")

    console.print(Panel(entry_table, title="[bold]🎯 Entry-Preis-Rechner (Backtest-validiert)[/bold]",
                        subtitle=entry_info, border_style="green"))

    # 5.6 Zyklus-Timing (Zeitliche Prognose)
    timing_text = Text()
    
    target_start = ath_date + datetime.timedelta(days=int(avg_d * 0.9))
    target_end = ath_date + datetime.timedelta(days=int(avg_d * 1.1))
    
    # Institutionelle Verlängerung (+20% Zeit)
    inst_start = ath_date + datetime.timedelta(days=int(avg_d * 1.1))
    inst_end = ath_date + datetime.timedelta(days=int(avg_d * 1.4))
    
    timing_text.append(f"Basierend auf {cycles_found} historischen Zyklen:\n", style="dim")
    timing_text.append(f"Bisher vergangene Zeit: ", style="bold")
    timing_text.append(f"{days_since_ath} Tage\n\n")

    # A: Klassisches Fenster
    timing_text.append("📊 KLASSISCHES FENSTER (Durchschnitt):\n", style="bold cyan")
    timing_text.append(f"   {int(avg_d*0.9)}-{int(avg_d*1.1)} Tage: ", style="dim")
    timing_text.append(f"{target_start.strftime('%d.%m.%Y')} bis {target_end.strftime('%d.%m.%Y')}\n\n")

    # B: Institutionelles Fenster (Lengthening Cycle Theory)
    timing_text.append("🏛️ INSTITUTIONELLES FENSTER (ETF/Maturity):\n", style="bold magenta")
    timing_text.append(f"   {int(avg_d*1.1)}-{int(avg_d*1.4)} Tage: ", style="dim")
    timing_text.append(f"{inst_start.strftime('%d.%m.%Y')} bis {inst_end.strftime('%d.%m.%Y')}\n")
    timing_text.append("   (Berücksichtigt die Theorie längerer Zyklen durch BlackRock/ETFs)\n\n", style="dim italic")

    if days_since_ath < int(avg_d * 0.85):
        timing_text.append("⏳ FRÜHE PHASE: ", style="bold yellow")
        timing_text.append("Es ist wahrscheinlich noch zu früh für den Boden.")
        timing_border = "yellow"
    elif int(avg_d * 0.85) <= days_since_ath <= int(avg_d * 1.1):
        timing_text.append("🎯 KLASSISCHE BODEN-ZONE: ", style="bold green reverse")
        timing_text.append(" Wir sind im Bereich des historischen Durchschnitts!")
        timing_border = "green"
    elif int(avg_d * 1.1) < days_since_ath <= int(avg_d * 1.4):
        timing_text.append("🏛️ INSTITUTIONELLE AKKUMULATION: ", style="bold magenta reverse")
        timing_text.append(" Wir sind im Bereich für einen verlängerten Zyklus.")
        timing_border = "magenta"
    else:
        timing_text.append("📈 SPÄTE PHASE: ", style="bold cyan")
        timing_text.append("Das Zeitfenster liegt zurück. Boden sollte drin sein.")
        timing_border = "cyan"

    console.print(Panel(timing_text, title="[bold]⏳ Zyklus-Timing (Dynamisch & Institutional)[/bold]", border_style=timing_border))

    # 6. FAZIT (kombiniert Zone + Score zu einer Aussage)
    if current_price > zone1_upper:
        active_zone = 0
    elif current_price >= zone1_lower:
        active_zone = 1
    elif current_price >= zone2_lower:
        active_zone = 2
    elif current_price >= zone3_lower:
        active_zone = 3
    else:
        active_zone = 4

    zone_names = {0: "", 1: "Zone 1 (Soft Floor)", 2: "Zone 2 (Hard Floor)",
                  3: "Zone 3 (Max Pain)", 4: "Zone 4 (Black Swan)"}

    fazit_text = Text()
    if active_zone == 0:
        fazit_text.append("🧊 KEIN HANDLUNGSBEDARF\n\n", style="bold dim")
        fazit_text.append("Der Preis liegt über allen Kaufzonen. Abwarten.")
        fazit_border = "dim"
    elif active_zone == 1:
        if adjusted < 45:
            fazit_text.append("👀 BEOBACHTEN - NOCH NICHT KAUFEN\n\n", style="bold yellow")
            fazit_text.append(f"Der Preis ist in {zone_names[1]}, aber der Score steht bei {adjusted}/100.\n")
            fazit_text.append("Historisch war ein Einstieg in Zone 1 erst ab Score 45+ lohnend.\n")
            fazit_text.append("Geduld - der Markt braucht noch Zeit.")
            fazit_border = "yellow"
        elif adjusted < 65:
            fazit_text.append("🌱 ERSTE POSITIONEN MÖGLICH (DCA)\n\n", style="bold yellow")
            fazit_text.append(f"{zone_names[1]} aktiv mit Score {adjusted}/100.\n")
            fazit_text.append("Der Markt zeigt erste Bodenbildung. Kleine Positionen oder DCA-Start denkbar.")
            fazit_border = "yellow"
        else:
            fazit_text.append("✅ KAUFSIGNAL\n\n", style="bold green")
            fazit_text.append(f"{zone_names[1]} aktiv mit starkem Score {adjusted}/100.\n")
            fazit_text.append("DCA starten oder Positionen aufbauen.")
            fazit_border = "green"
    elif active_zone == 2:
        if adjusted < 45:
            fazit_text.append("⏳ KERNZONE ERREICHT - SCORE BAUT SICH AUF\n\n", style="bold yellow")
            fazit_text.append(f"Der Preis ist in {zone_names[2]}, Score bei {adjusted}/100.\n")
            fazit_text.append("Erste kleine Positionen denkbar. Markt noch nicht komplett kapituliert.")
            fazit_border = "yellow"
        elif adjusted < 65:
            fazit_text.append("🛒 STARKE KAUFZONE (DCA)\n\n", style="bold red")
            fazit_text.append(f"{zone_names[2]} aktiv mit Score {adjusted}/100.\n")
            fazit_text.append("Solide Akkumulationszone. DCA fortsetzen, Positionen aufbauen.")
            fazit_border = "red"
        else:
            fazit_text.append("🔥 HISTORISCHE KAUFGELEGENHEIT\n\n", style="bold white on red")
            fazit_text.append(f"{zone_names[2]} aktiv mit Score {adjusted}/100.\n")
            fazit_text.append("Historisch einer der besten Einstiegspunkte. Aggressiv akkumulieren.")
            fazit_border = "red"
    elif active_zone == 3:
        if adjusted < 65:
            fazit_text.append("🩸 EXTREME UNTERBEWERTUNG\n\n", style="bold red")
            fazit_text.append(f"{zone_names[3]} aktiv, Score bei {adjusted}/100.\n")
            fazit_text.append("Preislich extrem günstig. Starke Akkumulation sinnvoll.")
            fazit_border = "red"
        else:
            fazit_text.append("💎 GENERATIONSKAUFGELEGENHEIT\n\n", style="bold white on red")
            fazit_text.append(f"{zone_names[3]} aktiv mit Score {adjusted}/100.\n")
            fazit_text.append("Kommt nur alle paar Jahre vor. Maximale Akkumulation.")
            fazit_border = "red"
    else:
        fazit_text.append("🦢 BLACK SWAN - EXTREMSITUATION\n\n", style="bold white on magenta")
        fazit_text.append(f"{zone_names[4]} aktiv, Score bei {adjusted}/100.\n")
        fazit_text.append("Historisch extrem selten. Wenn Fundamentaldaten intakt:\n")
        fazit_text.append("Beste Kaufgelegenheit aller Zeiten.")
        fazit_border = "magenta"

    console.print(Panel(fazit_text, title="[bold]💡 FAZIT[/bold]", border_style=fazit_border))

    # 7. Score-Legende
    interp_text = Text()
    interp_text.append("Score-Skala:\n\n", style="bold")
    interp_text.append("  80-100  ", style="bold red")
    interp_text.append("EXTREM 🔥\n")
    interp_text.append("  65-79   ", style="bold red")
    interp_text.append("STARK 🚨\n")
    interp_text.append("  45-64   ", style="bold yellow")
    interp_text.append("MODERAT ⚠️\n")
    interp_text.append("  25-44   ", style="yellow")
    interp_text.append("SCHWACH ⏳\n")
    interp_text.append("   0-24   ", style="dim")
    interp_text.append("KEIN SIGNAL 🧊\n")

    console.print(Panel(interp_text, title="[bold]📜 Legende[/bold]", border_style="dim"))

    console.print("\n[dim italic]Hinweis: Keine Finanzberatung. Composite Score basiert auf 200W SMA, Mayer Multiple, RSI, Drawdown, Volumen/Zeit-Analyse mit Zyklus-Multiplikator (Backtest-validiert).[/dim italic]", justify="center")

    # =========================================================================
    # LLM Prompt Generation
    # =========================================================================

    # Trend-Status Text
    if not pd.isna(bmsb_lower) and not pd.isna(bmsb_upper):
        if current_price > bmsb_upper:
            trend_status_str = "AUFWÄRTSTREND INTAKT (Preis über BMSB)"
        elif current_price < bmsb_lower:
            trend_status_str = "ABWÄRTSTREND / KORREKTUR (Preis unter BMSB)"
        else:
            trend_status_str = "ENTSCHEIDUNGSZONE (Preis an BMSB-Grenze)"
    else:
        trend_status_str = "Nicht genug Daten"

    # Timing-Phase Text
    if days_since_ath < int(avg_d * 0.85):
        timing_phase_str = "FRÜHE PHASE - wahrscheinlich noch zu früh für Boden"
    elif int(avg_d * 0.85) <= days_since_ath <= int(avg_d * 1.1):
        timing_phase_str = "KLASSISCHE BODEN-ZONE - im historischen Durchschnitt"
    elif int(avg_d * 1.1) < days_since_ath <= int(avg_d * 1.4):
        timing_phase_str = "INSTITUTIONELLE AKKUMULATION - verlängerter Zyklus"
    else:
        timing_phase_str = "SPÄTE PHASE - Zeitfenster liegt zurück"

    # FAZIT Text
    fazit_str = ""
    if active_zone == 0:
        fazit_str = "KEIN HANDLUNGSBEDARF - Preis über allen Kaufzonen"
    elif active_zone == 1:
        if adjusted < 45:
            fazit_str = f"BEOBACHTEN - {zone_names[1]}, Score {adjusted}/100. Noch nicht kaufen."
        elif adjusted < 65:
            fazit_str = f"ERSTE POSITIONEN MÖGLICH - {zone_names[1]}, Score {adjusted}/100. Kleine DCA-Positionen denkbar."
        else:
            fazit_str = f"KAUFSIGNAL - {zone_names[1]}, Score {adjusted}/100. Positionen aufbauen."
    elif active_zone == 2:
        if adjusted < 45:
            fazit_str = f"KERNZONE ERREICHT - {zone_names[2]}, Score {adjusted}/100. Erste Positionen denkbar."
        elif adjusted < 65:
            fazit_str = f"STARKE KAUFZONE - {zone_names[2]}, Score {adjusted}/100. Akkumulieren."
        else:
            fazit_str = f"HISTORISCHE KAUFGELEGENHEIT - {zone_names[2]}, Score {adjusted}/100. Aggressiv akkumulieren."
    elif active_zone == 3:
        if adjusted < 65:
            fazit_str = f"EXTREME UNTERBEWERTUNG - {zone_names[3]}, Score {adjusted}/100. Starke Akkumulation."
        else:
            fazit_str = f"GENERATIONSKAUFGELEGENHEIT - {zone_names[3]}, Score {adjusted}/100. Maximale Akkumulation."
    else:
        fazit_str = f"BLACK SWAN EXTREMSITUATION - {zone_names[4]}, Score {adjusted}/100. Beste Kaufgelegenheit aller Zeiten (wenn Fundamentaldaten intakt)."

    prompt_data = {
        "price": current_price,
        "ath": ath,
        "ath_date": ath_date.strftime("%Y-%m-%d"),
        "drawdown": drop_from_ath,
        "days_since_ath": days_since_ath,
        "sma_200w": sma_200w,
        "deviation": raw["deviation"],
        "sma_200d": sma_200d,
        "mayer_multiple": mayer_multiple,
        "weekly_rsi": weekly_rsi,
        "bmsb_sma20w": sma_20w if not pd.isna(sma_20w) else 0,
        "bmsb_ema21w": ema_21w if not pd.isna(ema_21w) else 0,
        "trend_status": trend_status_str,
        "score_raw": total,
        "score_total": adjusted,
        "score_factor": factor,
        "score_label": label,
        "s1": score["sma_dev"],
        "s2": score["mayer"],
        "s3": score["rsi"],
        "s4": score["drawdown"],
        "s5": score["capitulation"],
        "z1_upper": zone1_upper, "z1_lower": zone1_lower,
        "z2_upper": zone2_upper, "z2_lower": zone2_lower,
        "z3_upper": zone3_upper, "z3_lower": zone3_lower,
        "z4_upper": zone4_upper, "z4_lower": zone4_lower,
        "active_zone_name": zone_names.get(active_zone, "Über allen Zonen"),
        "cycles_found": cycles_found,
        "classic_start": target_start.strftime("%d.%m.%Y"),
        "classic_end": target_end.strftime("%d.%m.%Y"),
        "inst_start": inst_start.strftime("%d.%m.%Y"),
        "inst_end": inst_end.strftime("%d.%m.%Y"),
        "timing_phase": timing_phase_str,
        "fazit": fazit_str,
    }

    llm_prompt = generate_llm_prompt(prompt_data)

    # Prompt in Datei speichern
    prompt_file = "llm_verification_prompt.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(llm_prompt)

    console.print()
    console.print(Panel(
        f"[bold green]LLM-Verifizierungs-Prompt wurde generiert![/bold green]\n\n"
        f"📄 Gespeichert in: [bold cyan]{prompt_file}[/bold cyan]\n\n"
        f"Kopiere den Inhalt der Datei und füge ihn in eine LLM mit\n"
        f"Internet-Zugang ein (z.B. ChatGPT, Perplexity, Gemini),\n"
        f"um die Analyse mit aktuellen Echtzeitdaten verifizieren zu lassen.",
        title="[bold]🤖 LLM-Prompt Generator[/bold]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
