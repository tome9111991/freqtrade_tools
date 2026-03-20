"""
BTC Bottom Analyzer - Backtest
Vergleicht die alte Logik (btc_analyzer.py) mit der neuen Composite-Score Logik
gegen historische Bitcoin-Böden und Nicht-Böden.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


# ============================================================
# DATEN & HILFSFUNKTIONEN
# ============================================================

def load_data():
    console.print("[cyan]Lade BTC-Daten von Yahoo Finance...[/cyan]")
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="max")
    df.index = df.index.tz_localize(None)
    console.print(f"[green]{len(df)} Tage geladen ({df.index[0].strftime('%Y-%m-%d')} bis {df.index[-1].strftime('%Y-%m-%d')})[/green]\n")
    return df


def compute_rsi(series, period=14):
    """RSI nach Wilders Methode (EMA-basiert)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_sma(series, window):
    if len(series) < window:
        return np.nan
    return series.rolling(window=window).mean().iloc[-1]


def get_forward_returns(df, test_date):
    """Renditen nach dem Test-Datum."""
    future = df[df.index > test_date]
    price = df[df.index <= test_date]['Close'].iloc[-1]
    result = {}
    for months, label in [(3, "3M"), (6, "6M"), (12, "12M")]:
        target = test_date + datetime.timedelta(days=months * 30)
        fslice = future[future.index <= target]
        if len(fslice) > 0:
            result[f"{label}_ret"] = ((fslice['Close'].iloc[-1] - price) / price) * 100
        else:
            result[f"{label}_ret"] = None
    return result


# ============================================================
# ALTE ANALYSE (aktuelle btc_analyzer.py Logik)
# ============================================================

def old_analysis(df, test_date):
    data = df[df.index <= test_date].copy()
    if len(data) < 50:
        return None

    price = data['Close'].iloc[-1]
    ath = data['High'].max()
    ath_date = data[data['High'] == ath].index[0]
    drawdown = ((price - ath) / ath) * 100

    weekly = data['Close'].resample('W').last().dropna()
    sma_200w = get_sma(weekly, 200)
    sma_200d = get_sma(data['Close'], 200)

    if pd.isna(sma_200w):
        sma_200w = weekly.mean()
    if pd.isna(sma_200d):
        sma_200d = data['Close'].mean()

    mayer = price / sma_200d if sma_200d > 0 else 1.0

    # --- Zonen (fest an 200W SMA) ---
    z1_lo, z1_hi = sma_200w * 1.10, sma_200w * 1.25
    z2_lo, z2_hi = sma_200w * 0.95, sma_200w * 1.10
    z3_lo, z3_hi = sma_200w * 0.80, sma_200w * 0.95
    z4_lo, z4_hi = sma_200w * 0.60, sma_200w * 0.80

    zone_score = 0
    if price < z4_lo:
        zone_score = 35
    elif z4_lo <= price <= z4_hi:
        zone_score = 35
    elif z3_lo <= price <= z3_hi:
        zone_score = 30
    elif z2_lo <= price <= z2_hi:
        zone_score = 25
    elif z1_lo <= price <= z1_hi:
        zone_score = 15

    # --- Timing (hardcoded 383 Tage) ---
    days_since_ath = (test_date - ath_date).days
    cycle_pct = (days_since_ath / 383) * 100

    timing_score = 0
    if 85 <= cycle_pct <= 150:
        timing_score = 20
    elif 50 <= cycle_pct < 85:
        timing_score = 10

    # --- BMSB ---
    sma_20w = get_sma(weekly, 20)
    ema_21w = weekly.ewm(span=21, adjust=False).mean().iloc[-1] if len(weekly) >= 21 else np.nan

    bmsb_score = 0
    if not pd.isna(sma_20w) and not pd.isna(ema_21w):
        bmsb_lower = min(sma_20w, ema_21w)
        if price < bmsb_lower:
            bmsb_score = 15
        elif price <= max(sma_20w, ema_21w):
            bmsb_score = 10

    # --- Drawdown Score ---
    dd_score = 0
    if drawdown <= -80: dd_score = 20
    elif drawdown <= -70: dd_score = 15
    elif drawdown <= -50: dd_score = 10
    elif drawdown <= -30: dd_score = 5

    # --- Kapitulation (14-Tage Fenster) ---
    cap_score = 0
    if len(data) >= 90:
        recent = data.tail(14)
        avg_vol = data['Volume'].tail(90).mean()
        for _, row in recent.iterrows():
            if avg_vol > 0 and row['Volume'] > avg_vol * 1.8:
                rng = row['High'] - row['Low']
                if rng > 0:
                    wick = (row['Close'] - row['Low']) / rng
                    if wick > 0.4 and row['Close'] <= row['Open'] * 1.02:
                        cap_score = 10
                        break

    total = min(zone_score + timing_score + bmsb_score + dd_score + cap_score, 100)

    return {
        "score": total,
        "price": price,
        "drawdown": drawdown,
    }


# ============================================================
# NEUE ANALYSE (Composite Bottom Score)
# ============================================================

def new_analysis(df, test_date):
    data = df[df.index <= test_date].copy()
    if len(data) < 50:
        return None

    price = data['Close'].iloc[-1]
    ath = data['High'].max()
    ath_date = data[data['High'] == ath].index[0]
    drawdown = ((price - ath) / ath) * 100
    days_since_ath = (test_date - ath_date).days

    weekly = data['Close'].resample('W').last().dropna()
    sma_200w = get_sma(weekly, 200)
    sma_200d = get_sma(data['Close'], 200)

    if pd.isna(sma_200w):
        sma_200w = weekly.mean()
    if pd.isna(sma_200d):
        sma_200d = data['Close'].mean()

    mayer = price / sma_200d if sma_200d > 0 else 1.0
    deviation = ((price - sma_200w) / sma_200w) * 100 if sma_200w > 0 else 0

    # Weekly RSI
    weekly_rsi_series = compute_rsi(weekly, 14)
    rsi = weekly_rsi_series.iloc[-1] if len(weekly_rsi_series) > 0 and not pd.isna(weekly_rsi_series.iloc[-1]) else 50

    # ===== SCORING =====

    # 1. 200W SMA Abweichung (0-30)
    #    "An der 200W SMA" (±5%) ist historisch bereits ein starkes Boden-Signal
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

    # 3. Woechentlicher RSI (0-20)
    #    Leicht angepasste Grenzen um Randwerte besser zu erfassen
    if rsi <= 28:   s3 = 20
    elif rsi <= 33: s3 = 16
    elif rsi <= 38: s3 = 12
    elif rsi <= 43: s3 = 7
    elif rsi <= 50: s3 = 3
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
    if len(data) >= 90:
        avg_vol = data['Volume'].tail(90).mean()

        # a) Volumen-Spike in letzten 30 Tagen (ohne Docht-Anforderung,
        #    da Crashs oft ueber mehrere Tage laufen)
        max_vol_30d = data['Volume'].tail(30).max()
        if avg_vol > 0 and max_vol_30d > avg_vol * 2.0:
            s5 += 4

        # b) Volumen-Austrocknung (Selling Exhaustion) letzte 7 Tage
        if len(data) >= 7:
            vol_7d = data['Volume'].tail(7).mean()
            if avg_vol > 0 and vol_7d < avg_vol * 0.75:
                s5 += 3

        # c) Volatilitaets-Kompression (Ruhe nach dem Sturm)
        if len(data) >= 90:
            vol_short = data['Close'].tail(14).pct_change().std()
            vol_long = data['Close'].tail(90).pct_change().std()
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

    return {
        "score": total,
        "price": price,
        "drawdown": drawdown,
        "deviation": deviation,
        "mayer": mayer,
        "rsi": rsi,
        "components": {"SMA": s1, "Mayer": s2, "RSI": s3, "DD": s4, "Cap": s5},
    }


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    df = load_data()

    test_cases = [
        # (Datum, Label, Erwartung)
        # --- Echte Boeden ---
        ("2015-01-14", "BODEN 2015",              "bottom"),
        ("2018-12-15", "BODEN 2018",              "bottom"),
        ("2022-11-21", "BODEN 2022",              "bottom"),
        # --- 30 Tage VOR dem Boden ---
        ("2018-11-15", "30d vor Boden 2018",      "near"),
        ("2022-10-22", "30d vor Boden 2022",      "near"),
        # --- Kein Boden (schmerzhaft, aber zu frueh) ---
        ("2018-02-06", "Erster Crash 2018",        "false"),
        ("2021-05-19", "Mid-Cycle Crash 2021",     "false"),
        ("2022-06-18", "Luna Crash",               "false"),
        # --- Bullenmarkt (Score muss niedrig sein) ---
        ("2021-03-15", "Bullenmarkt Maerz 2021",  "bull"),
        ("2024-03-15", "Bullenmarkt Maerz 2024",  "bull"),
    ]

    # =========================================
    # TABELLE 1: Hauptvergleich ALT vs NEU
    # =========================================
    t1 = Table(
        title="BACKTEST: Alte Logik vs. Neuer Composite Score",
        box=box.DOUBLE_EDGE, show_lines=True, expand=True,
    )
    t1.add_column("Datum", width=12, style="bold")
    t1.add_column("Beschreibung", width=26)
    t1.add_column("Preis", justify="right", width=10)
    t1.add_column("DD%", justify="right", width=7)
    t1.add_column("ALT", justify="center", width=5, style="yellow")
    t1.add_column("NEU", justify="center", width=5, style="cyan")
    t1.add_column("+/-", justify="center", width=5)
    t1.add_column("6M Ret", justify="right", width=9)
    t1.add_column("12M Ret", justify="right", width=9)
    t1.add_column("Bewertung", width=16)

    results = []

    for date_str, label, expect in test_cases:
        td = pd.Timestamp(date_str)
        if td > df.index[-1] or td < df.index[0]:
            continue

        old = old_analysis(df, td)
        new = new_analysis(df, td)
        fwd = get_forward_returns(df, td)

        if old is None or new is None:
            continue

        results.append((date_str, label, expect, old, new, fwd))

        delta = new["score"] - old["score"]
        ds = f"[green]+{delta}[/green]" if delta > 0 else f"[red]{delta}[/red]" if delta < 0 else "="

        def fmt_ret(val):
            if val is None: return "[dim]N/A[/dim]"
            return f"[green]+{val:.0f}%[/green]" if val >= 0 else f"[red]{val:.0f}%[/red]"

        # Bewertung des NEUEN Scores
        ns = new["score"]
        if expect == "bottom":
            bew = "[bold green]TREFFER[/bold green]" if ns >= 65 else "[yellow]TEILWEISE[/yellow]" if ns >= 45 else "[red]VERPASST[/red]"
        elif expect == "near":
            bew = "[green]SIGNAL[/green]" if ns >= 40 else "[red]ZU SPAET[/red]"
        elif expect == "bull":
            bew = "[bold green]KORREKT[/bold green]" if ns <= 20 else "[red]FEHLALARM[/red]"
        else:  # false
            bew = "[green]KORREKT[/green]" if ns < 65 else "[yellow]GRENZFALL[/yellow]"

        sty_old = "bold green" if old["score"] >= 65 else "yellow" if old["score"] >= 40 else "dim"
        sty_new = "bold green" if ns >= 65 else "yellow" if ns >= 40 else "dim"

        t1.add_row(
            date_str, label,
            f"${old['price']:,.0f}",
            f"{old['drawdown']:.0f}%",
            f"[{sty_old}]{old['score']}[/{sty_old}]",
            f"[{sty_new}]{ns}[/{sty_new}]",
            ds,
            fmt_ret(fwd.get("6M_ret")),
            fmt_ret(fwd.get("12M_ret")),
            bew,
        )

    console.print(t1)

    # =========================================
    # TABELLE 2: Score-Aufschluesselung NEU
    # =========================================
    t2 = Table(
        title="Score-Aufschluesselung (Neuer Composite Score)",
        box=box.SIMPLE_HEAD, show_lines=True,
    )
    t2.add_column("Datum", width=12)
    t2.add_column("Label", width=26)
    t2.add_column("200W\n(0-30)", justify="center", width=7)
    t2.add_column("Mayer\n(0-20)", justify="center", width=7)
    t2.add_column("RSI\n(0-20)", justify="center", width=7)
    t2.add_column("DD\n(0-20)", justify="center", width=7)
    t2.add_column("Cap\n(0-10)", justify="center", width=7)
    t2.add_column("TOTAL", justify="center", style="bold", width=7)

    for date_str, label, expect, old, new, fwd in results:
        c = new["components"]
        t2.add_row(
            date_str, label,
            str(c["SMA"]), str(c["Mayer"]), str(c["RSI"]),
            str(c["DD"]), str(c["Cap"]),
            str(new["score"]),
        )

    console.print("\n")
    console.print(t2)

    # =========================================
    # TABELLE 3: Rohdaten
    # =========================================
    t3 = Table(
        title="Indikator-Rohdaten",
        box=box.SIMPLE_HEAD, show_lines=True,
    )
    t3.add_column("Datum", width=12)
    t3.add_column("Preis", justify="right", width=10)
    t3.add_column("200W Dev%", justify="right", width=10)
    t3.add_column("Mayer", justify="right", width=7)
    t3.add_column("W-RSI", justify="right", width=7)
    t3.add_column("DD%", justify="right", width=7)

    for date_str, label, expect, old, new, fwd in results:
        t3.add_row(
            date_str,
            f"${new['price']:,.0f}",
            f"{new['deviation']:.1f}%",
            f"{new['mayer']:.2f}",
            f"{new['rsi']:.1f}",
            f"{new['drawdown']:.1f}%",
        )

    console.print("\n")
    console.print(t3)

    # =========================================
    # ZUSAMMENFASSUNG
    # =========================================
    def avg_scores(results, expect_type, idx):
        scores = [r[idx]["score"] for r in results if r[2] == expect_type]
        return np.mean(scores) if scores else 0

    ob = avg_scores(results, "bottom", 3)
    nb = avg_scores(results, "bottom", 4)
    obu = avg_scores(results, "bull", 3)
    nbu = avg_scores(results, "bull", 4)
    of_ = avg_scores(results, "false", 3)
    nf = avg_scores(results, "false", 4)

    summary = f"""Durchschnitt Echte Boeden:     ALT {ob:.0f}  ->  NEU {nb:.0f}
Durchschnitt Bullenmarkt:     ALT {obu:.0f}  ->  NEU {nbu:.0f}
Durchschnitt Fehlalarme:      ALT {of_:.0f}  ->  NEU {nf:.0f}

Spread (Boden - Bullenmarkt): ALT {ob - obu:.0f}  ->  NEU {nb - nbu:.0f}
Spread (Boden - Fehlalarm):   ALT {ob - of_:.0f}  ->  NEU {nb - nf:.0f}"""

    console.print("\n")
    console.print(Panel(summary, title="ZUSAMMENFASSUNG", border_style="cyan"))
    console.print("\n[dim]Score: 0-20 kein Signal | 20-40 schwach | 40-60 moderat | 60-80 stark | 80-100 extrem[/dim]")


if __name__ == "__main__":
    main()
