import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import sys
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


def compute_bottom_score(price, sma_200w, sma_200d, weekly_rsi, ath, days_since_ath, df_tail_90, df_tail_30, df_tail_14, df_tail_7):
    """
    Berechnet den Composite Bottom Score (0-100).

    Komponenten:
      1. 200W SMA Abweichung (0-30)
      2. Mayer Multiple (0-20)
      3. Woechentlicher RSI (0-20)
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

    # 3. Woechentlicher RSI (0-20)
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

        # c) Volatilitaets-Kompression
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

    return {
        "total": total,
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
        },
    }


def score_label(score):
    if score >= 80: return ("EXTREM", "bold white on red")
    elif score >= 65: return ("STARK", "bold red")
    elif score >= 45: return ("MODERAT", "bold yellow")
    elif score >= 25: return ("SCHWACH", "yellow")
    else: return ("KEIN SIGNAL", "dim")


def main():
    console = Console()

    with console.status("[bold cyan]Lade historische und aktuelle Marktdaten von Yahoo Finance...", spinner="dots"):
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="max")

        if df.empty:
            console.print("[bold red]Fehler: Konnte keine Daten laden.[/bold red]")
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

        # --- COMPOSITE BOTTOM SCORE ---
        tail_90 = df.tail(90) if len(df) >= 90 else None
        tail_30 = df.tail(30) if len(df) >= 30 else None
        tail_14 = df.tail(14) if len(df) >= 14 else None
        tail_7 = df.tail(7) if len(df) >= 7 else None

        score = compute_bottom_score(
            current_price, sma_200w, sma_200d, weekly_rsi, ath,
            days_since_ath, tail_90, tail_30, tail_14, tail_7,
        )

    # =========================================================================
    # CLI UI
    # =========================================================================

    # 1. Header
    header = Panel(
        Text("BITCOIN BOTTOM ANALYZER", justify="center", style="bold white on blue"),
        box=box.DOUBLE_EDGE,
        border_style="blue",
        padding=(1, 1)
    )
    console.print(header)

    # 2. COMPOSITE BOTTOM SCORE (Herzstück)
    total = score["total"]
    label, label_style = score_label(total)
    raw = score["raw"]

    score_text = Text()
    score_text.append(f"\n  BOTTOM SCORE:  ", style="bold")
    score_text.append(f" {total} / 100 ", style=f"bold white on {'red' if total >= 65 else 'yellow' if total >= 40 else 'green'}")
    score_text.append(f"  {label}\n\n", style=label_style)

    # Balken-Visualisierung
    bar_filled = int(total / 2)  # 0-50 Zeichen
    bar_empty = 50 - bar_filled
    bar_color = "red" if total >= 65 else "yellow" if total >= 40 else "green"
    score_text.append(f"  [{'#' * bar_filled}{'.' * bar_empty}]\n\n", style=bar_color)

    # Komponenten-Aufschluesselung mit Klartext
    dev = raw['deviation']
    mayer_val = raw['mayer_val']
    rsi_val = raw['rsi_val']
    dd_val = raw['drawdown_val']

    # Dynamische Erklaerungen je nach aktuellem Wert
    if dev <= 0:
        sma_explain = f"Preis liegt {abs(dev):.0f}% UNTER dem Langzeit-Durchschnitt. Historisch guenstig!"
    elif dev <= 15:
        sma_explain = f"Preis liegt {dev:.0f}% ueber dem Langzeit-Durchschnitt. Nahe dran, aber noch nicht drunter."
    else:
        sma_explain = f"Preis liegt {dev:.0f}% ueber dem Langzeit-Durchschnitt. Noch weit von einer Kaufzone entfernt."

    if mayer_val <= 0.8:
        mayer_explain = f"BTC kostet nur {mayer_val:.2f}x vom Halbjahres-Trend. Unter 0.8 = historisch billig!"
    elif mayer_val <= 1.2:
        mayer_explain = f"BTC kostet {mayer_val:.2f}x vom Halbjahres-Trend. Fair bewertet, noch kein Schnaeppchen."
    else:
        mayer_explain = f"BTC kostet {mayer_val:.2f}x vom Halbjahres-Trend. Ueberhitzt/teuer."

    if rsi_val <= 33:
        rsi_explain = f"Markt ist stark ueberverkauft (RSI {rsi_val:.0f}). Verkaeufer werden muede!"
    elif rsi_val <= 45:
        rsi_explain = f"Markt zeigt Schwaeche (RSI {rsi_val:.0f}). Noch nicht am Limit."
    else:
        rsi_explain = f"Markt hat noch Kraft (RSI {rsi_val:.0f}). Kein Ueberverkauf-Signal."

    if dd_val <= -75:
        dd_explain = f"Preis ist {abs(dd_val):.0f}% unter dem Allzeithoch. Extreme Crash-Tiefe!"
    elif dd_val <= -50:
        dd_explain = f"Preis ist {abs(dd_val):.0f}% unter dem Allzeithoch. Tiefer Baerenmarkt."
    elif dd_val <= -30:
        dd_explain = f"Preis ist {abs(dd_val):.0f}% unter dem Allzeithoch. Korrektur, aber noch nicht tief genug."
    else:
        dd_explain = f"Preis ist nur {abs(dd_val):.0f}% unter dem Allzeithoch. Kein tiefer Crash."

    cap_pts = score["capitulation"]
    if cap_pts >= 7:
        cap_explain = "Zeichen von Massenpanik erkannt! Hohes Volumen + Markt beruhigt sich."
    elif cap_pts >= 4:
        cap_explain = "Kapitulations-Signale erkannt (Volumen/Zeit). Markt zeigt Erschoepfung."
    else:
        cap_explain = "Keine Massenpanik erkannt. Markt blutet langsam, kein grosser Ausverkauf."

    components = [
        ("Langzeit-Trend",   score["sma_dev"],      30, sma_explain),
        ("Unterbewertung",   score["mayer"],         20, mayer_explain),
        ("Verkaufsdruck",    score["rsi"],           20, rsi_explain),
        ("Crash-Tiefe",      score["drawdown"],      20, dd_explain),
        ("Panik-Verkauf",    score["capitulation"],  10, cap_explain),
    ]

    for name, pts, max_pts, explain in components:
        pct = pts / max_pts if max_pts > 0 else 0
        color = "red" if pct >= 0.7 else "yellow" if pct >= 0.4 else "dim"
        bar_w = 15
        filled = int(pct * bar_w)
        score_text.append(f"  {name:<18} ", style="bold")
        score_text.append(f"{'|' * filled}{'.' * (bar_w - filled)} ", style=color)
        score_text.append(f"{pts:>2}/{max_pts}\n", style=color)
        score_text.append(f"  {'':18} {explain}\n", style="dim italic")

    score_text.append(f"\n")
    border = "red" if total >= 65 else "yellow" if total >= 40 else "green"
    console.print(Panel(score_text, title="[bold]COMPOSITE BOTTOM SCORE[/bold]", border_style=border))

    # 3. Aktuelle Marktübersicht
    market_table = Table(show_header=False, box=box.SIMPLE, expand=True, border_style="cyan")
    market_table.add_column("Metrik", style="bold")
    market_table.add_column("Wert")

    market_table.add_row("Aktueller Preis", f"[bold green]${current_price:,.2f}[/bold green]")
    market_table.add_row("Hoechster Preis aller Zeiten", f"${ath:,.2f} [dim](am {ath_date.strftime('%Y-%m-%d')})[/dim]")
    market_table.add_row("Abstand zum Hoechststand", f"[{'red' if drop_from_ath < -30 else 'yellow' if drop_from_ath < 0 else 'green'}]{drop_from_ath:.2f}%[/{'red' if drop_from_ath < -30 else 'yellow' if drop_from_ath < 0 else 'green'}]")
    market_table.add_row("Langzeit-Durchschnitt (4 Jahre)", f"[yellow]${sma_200w:,.2f}[/yellow]")
    market_table.add_row("Halbjahres-Durchschnitt", f"[yellow]${sma_200d:,.2f}[/yellow]")
    market_table.add_row("Tage seit Hoechststand", f"{days_since_ath}")

    console.print(Panel(market_table, title="[bold]Aktuelle Marktdaten[/bold]", border_style="cyan"))

    # 4. Makro-Trend Status (BMSB)
    trend_text = Text()
    if not pd.isna(bmsb_lower) and not pd.isna(bmsb_upper):
        trend_text.append(f"Trend-Grenze liegt bei: ${bmsb_lower:,.0f} - ${bmsb_upper:,.0f}\n\n")

        if current_price > bmsb_upper:
            trend_text.append("AUFWAERTSTREND INTAKT\n", style="bold green")
            trend_text.append("Der Preis haelt sich ueber der Trend-Grenze. Solange das so bleibt,\n")
            trend_text.append("ist der uebergeordnete Trend positiv. Bodensuche ist hier zweitrangig.")
            trend_color = "green"
        elif current_price < bmsb_lower:
            trend_text.append("ABWAERTSTREND / KORREKTUR\n", style="bold red")
            trend_text.append("Der Preis ist unter die Trend-Grenze gefallen.\n")
            trend_text.append("Das bedeutet: Der Markt ist im Abwaertstrend. Jetzt wird der Bottom Score\n")
            trend_text.append("oben relevant - er zeigt dir, wie nah wir am moeglichen Boden sind.")
            trend_color = "red"
        else:
            trend_text.append("ENTSCHEIDUNGSZONE\n", style="bold yellow")
            trend_text.append("Der Preis kaempft genau an der Trend-Grenze.\n")
            trend_text.append("Hier entscheidet sich, ob der Markt wieder nach oben dreht oder weiter faellt.")
            trend_color = "yellow"
    else:
        trend_text.append("Nicht genug historische Daten fuer Trend-Analyse.", style="dim")
        trend_color = "dim"

    console.print(Panel(trend_text, title="[bold]Makro-Trend Status[/bold]", border_style=trend_color))

    # 5. Kaufzonen Radar
    zones_table = Table(box=box.MINIMAL_DOUBLE_HEAD, expand=True)
    zones_table.add_column("Zone", style="bold")
    zones_table.add_column("Preisbereich", justify="right")
    zones_table.add_column("Bedeutung")
    zones_table.add_column("Status", justify="center")

    def get_status(lower, upper, current, zone_num):
        if lower < current <= upper:
            return f"[bold green reverse] AKTIV (ZONE {zone_num}) [/bold green reverse]"
        return "[dim]-[/dim]"

    is_zone4 = current_price <= zone4_upper

    zones_table.add_row(
        "[blue]Zone 1 (Soft Floor)[/blue]",
        f"${zone1_lower:,.0f} - ${zone1_upper:,.0f}",
        "Erste Preisunterstuetzung",
        get_status(zone1_lower, zone1_upper, current_price, 1)
    )
    zones_table.add_row(
        "[yellow]Zone 2 (Hard Floor)[/yellow]",
        f"${zone2_lower:,.0f} - ${zone2_upper:,.0f}",
        "Historischer Kernbereich",
        get_status(zone2_lower, zone2_upper, current_price, 2)
    )
    zones_table.add_row(
        "[red]Zone 3 (Max Pain)[/red]",
        f"${zone3_lower:,.0f} - ${zone3_upper:,.0f}",
        "Stark unterbewerteter Bereich",
        get_status(zone3_lower, zone3_upper, current_price, 3)
    )
    zones_table.add_row(
        "[bold magenta]Zone 4 (Black Swan)[/bold magenta]",
        f"${zone4_lower:,.0f} - ${zone4_upper:,.0f}",
        "Historisch extrem selten",
        "[bold magenta reverse] AKTIV (ZONE 4) [/bold magenta reverse]" if is_zone4 else "[dim]-[/dim]"
    )

    console.print(Panel(zones_table, title="[bold]Kaufzonen-Radar (200W SMA)[/bold]", border_style="green"))

    if current_price > zone1_upper:
        console.print("[bold red]>>> AKTUELLER PREIS IST UEBER ALLEN KAUFZONEN <<<[/bold red]", justify="center")

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
        fazit_text.append("KEIN HANDLUNGSBEDARF\n\n", style="bold dim")
        fazit_text.append("Der Preis liegt ueber allen Kaufzonen. Abwarten.")
        fazit_border = "dim"
    elif active_zone == 1:
        if total < 45:
            fazit_text.append("BEOBACHTEN - NOCH NICHT KAUFEN\n\n", style="bold yellow")
            fazit_text.append(f"Der Preis ist in {zone_names[1]}, aber der Score steht bei {total}/100.\n")
            fazit_text.append("Historisch war ein Einstieg in Zone 1 erst ab Score 45+ lohnend.\n")
            fazit_text.append("Geduld - der Markt braucht noch Zeit.")
            fazit_border = "yellow"
        elif total < 65:
            fazit_text.append("ERSTE POSITIONEN MOEGLICH\n\n", style="bold yellow")
            fazit_text.append(f"{zone_names[1]} aktiv mit Score {total}/100.\n")
            fazit_text.append("Der Markt zeigt erste Bodenbildung. Kleine Positionen oder DCA-Start denkbar.")
            fazit_border = "yellow"
        else:
            fazit_text.append("KAUFSIGNAL\n\n", style="bold green")
            fazit_text.append(f"{zone_names[1]} aktiv mit starkem Score {total}/100.\n")
            fazit_text.append("DCA starten oder Positionen aufbauen.")
            fazit_border = "green"
    elif active_zone == 2:
        if total < 45:
            fazit_text.append("KERNZONE ERREICHT - SCORE BAUT SICH AUF\n\n", style="bold yellow")
            fazit_text.append(f"Der Preis ist in {zone_names[2]}, Score bei {total}/100.\n")
            fazit_text.append("Erste kleine Positionen denkbar. Markt noch nicht komplett kapituliert.")
            fazit_border = "yellow"
        elif total < 65:
            fazit_text.append("STARKE KAUFZONE\n\n", style="bold red")
            fazit_text.append(f"{zone_names[2]} aktiv mit Score {total}/100.\n")
            fazit_text.append("Solide Akkumulationszone. Positionen aufbauen.")
            fazit_border = "red"
        else:
            fazit_text.append("HISTORISCHE KAUFGELEGENHEIT\n\n", style="bold white on red")
            fazit_text.append(f"{zone_names[2]} aktiv mit Score {total}/100.\n")
            fazit_text.append("Historisch einer der besten Einstiegspunkte. Aggressiv akkumulieren.")
            fazit_border = "red"
    elif active_zone == 3:
        if total < 65:
            fazit_text.append("EXTREME UNTERBEWERTUNG\n\n", style="bold red")
            fazit_text.append(f"{zone_names[3]} aktiv, Score bei {total}/100.\n")
            fazit_text.append("Preislich extrem guenstig. Starke Akkumulation sinnvoll.")
            fazit_border = "red"
        else:
            fazit_text.append("GENERATIONSKAUFGELEGENHEIT\n\n", style="bold white on red")
            fazit_text.append(f"{zone_names[3]} aktiv mit Score {total}/100.\n")
            fazit_text.append("Kommt nur alle paar Jahre vor. Maximale Akkumulation.")
            fazit_border = "red"
    else:
        fazit_text.append("BLACK SWAN - EXTREMSITUATION\n\n", style="bold white on magenta")
        fazit_text.append(f"{zone_names[4]} aktiv, Score bei {total}/100.\n")
        fazit_text.append("Historisch extrem selten. Wenn Fundamentaldaten intakt:\n")
        fazit_text.append("Beste Kaufgelegenheit aller Zeiten.")
        fazit_border = "magenta"

    console.print(Panel(fazit_text, title="[bold]FAZIT[/bold]", border_style=fazit_border))

    # 7. Score-Legende
    interp_text = Text()
    interp_text.append("Score-Skala:\n\n", style="bold")
    interp_text.append("  80-100  ", style="bold red")
    interp_text.append("EXTREM\n")
    interp_text.append("  65-79   ", style="bold red")
    interp_text.append("STARK\n")
    interp_text.append("  45-64   ", style="bold yellow")
    interp_text.append("MODERAT\n")
    interp_text.append("  25-44   ", style="yellow")
    interp_text.append("SCHWACH\n")
    interp_text.append("   0-24   ", style="dim")
    interp_text.append("KEIN SIGNAL\n")

    console.print(Panel(interp_text, title="[bold]Legende[/bold]", border_style="dim"))

    console.print("\n[dim italic]Hinweis: Keine Finanzberatung. Composite Score basiert auf 200W SMA, Mayer Multiple, RSI, Drawdown, Volumen- und Zeit-Analyse.[/dim italic]", justify="center")


if __name__ == "__main__":
    main()
