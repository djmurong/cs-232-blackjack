"""
Blackjack Monte Carlo Simulation
MATH 242 / CS 232 — Duke University, Spring 2026

Same assumptions as the analytical model:
  - Ace = 1 always
  - Infinite deck: P(Ace) = 1/13, P(2-9) = 1/13, P(10-value) = 4/13
  - Player: hit if total < 17, stand if total >= 17
  - Dealer: same rule
  - Blackjack: Ace + 10-value on opening two cards only
"""

import random
import numpy as np

# ── Card draw ─────────────────────────────────────────────────────
def draw():
    """Sample one card. Returns value 1-10 (Ace=1, 10-value=10)."""
    return min(random.randint(1, 13), 10)

# ── Single hand simulation ────────────────────────────────────────
def play_hand(payout_bj=1.5):
    """
    Simulate one hand. Returns (profit, state_sequence).

    States visited:
      0=Start, 1=Hit, 2=Stand, 3=Bust, 4=BJ
      5=Win,   6=Lose, 7=Push, 8=BJ_Win
    """
    path = [0]  # Start

    c1, c2 = draw(), draw()
    is_bj = (c1 == 1 and c2 == 10) or (c1 == 10 and c2 == 1)

    # ── Player blackjack ──────────────────────────────────────────
    if is_bj:
        path.append(4)  # BJ state
        d1, d2 = draw(), draw()
        dealer_bj = (d1 == 1 and d2 == 10) or (d1 == 10 and d2 == 1)
        if dealer_bj:
            path.append(7)   # Push
            return 0.0, path
        else:
            path.append(8)   # BJ Win
            return payout_bj, path

    # ── Player hits ───────────────────────────────────────────────
    total = c1 + c2
    if total < 17:
        path.append(1)       # Hit
        while total < 17:
            total += draw()
        if total > 21:
            path.append(3)   # Bust
            path.append(6)   # Lose
            return -1.0, path

    path.append(2)           # Stand

    # ── Dealer plays ──────────────────────────────────────────────
    d1, d2 = draw(), draw()
    dealer_bj = (d1 == 1 and d2 == 10) or (d1 == 10 and d2 == 1)
    if dealer_bj:
        path.append(6)       # Lose
        return -1.0, path

    dtotal = d1 + d2
    while dtotal < 17:
        dtotal += draw()

    # ── Compare ───────────────────────────────────────────────────
    if dtotal > 21 or total > dtotal:
        path.append(5); return  1.0, path
    elif total == dtotal:
        path.append(7); return  0.0, path
    else:
        path.append(6); return -1.0, path

# ── Simulation ────────────────────────────────────────────────────
def simulate(n=1_000_000, payout_bj=1.5, seed=42):
    """
    Run n hands and return:
      ev          : average profit per hand
      state_freq  : empirical visit frequency for each of 9 states
      outcome_counts : {Win, Lose, Push, BJ_Win} counts
    """
    random.seed(seed)

    total_profit  = 0.0
    state_counts  = np.zeros(9)
    outcome_counts = {'Win': 0, 'Lose': 0, 'Push': 0, 'BJ_Win': 0}
    outcome_map   = {5: 'Win', 6: 'Lose', 7: 'Push', 8: 'BJ_Win'}

    for _ in range(n):
        profit, path = play_hand(payout_bj)
        total_profit += profit
        for s in path:
            state_counts[s] += 1
        final = path[-1]
        if final in outcome_map:
            outcome_counts[outcome_map[final]] += 1

    ev         = total_profit / n
    state_freq = state_counts / state_counts.sum()
    return ev, state_freq, outcome_counts

# ── Run and print ─────────────────────────────────────────────────
STATE_NAMES = ['Start','Hit','Stand','Bust','BJ','Win','Lose','Push','BJ_Win']
N = 1_000_000

print(f"Running {N:,} simulated hands...\n")

for label, payout in [("3:2  (payout = 1.5)", 1.5), ("6:5  (payout = 1.2)", 1.2)]:
    ev, freq, counts = simulate(n=N, payout_bj=payout)

    print(f"{'─'*50}")
    print(f"  Rule: {label}")
    print(f"{'─'*50}")
    print(f"  EV per hand     = {ev:+.6f}")
    print(f"  House edge      = {-ev*100:.4f}%\n")

    total_hands = sum(counts.values())
    print(f"  Outcome counts (out of {total_hands:,} hands):")
    for name, count in counts.items():
        print(f"    {name:8s}: {count:7,}  ({count/total_hands*100:.2f}%)")

    print(f"\n  State visit frequencies:")
    for name, f in zip(STATE_NAMES, freq):
        bar = '█' * int(f * 40)
        print(f"    {name:8s}: {f:.6f}  {bar}")
    print()

print(f"{'─'*50}")
ev_32, _, _ = simulate(n=N, payout_bj=1.5, seed=42)
ev_65, _, _ = simulate(n=N, payout_bj=1.2, seed=42)
print(f"  Delta EV (3:2 -> 6:5) = {ev_65 - ev_32:+.6f}")
print(f"  House edge increase   = {(ev_32 - ev_65)*100:.4f} pp")