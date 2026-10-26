"""
PitchLens API - explainable match intelligence from synthetic football events.
Run:  export ANTHROPIC_API_KEY=sk-ant-...   (optional: falls back to template text)
      uvicorn pitchlens_api:app --reload --port 8000

Pipeline: (a) ingest  -> generate_match() / POST /autoevent/ingest
          (b) interpret -> compute_state(), build_momentum()
          (c) explain -> detect_moments()  (every moment carries computed evidence)
          (d) render -> /overlays, /stream  (timed, machine-readable overlay cards)
          (e) personalize -> persona, language, favourite club, player focus (Claude)
"""
import os, re, json, math, random, asyncio
import requests
from xml.sax.saxutils import escape as xml_escape
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:  # loads ANTHROPIC_API_KEY from a .env file next to this script
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import anthropic
    CLIENT = anthropic.Anthropic() if os.getenv("ANTHROPIC_API_KEY") else None
except Exception:
    CLIENT = None
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

LANGS = {"en": "English", "fr": "French", "pt": "Portuguese", "yo": "Yoruba", "ha": "Hausa",
         "ig": "Igbo", "pcm": "Nigerian Pidgin"}
# Azure AI Speech (voice-over). Set AZURE_SPEECH_KEY + AZURE_SPEECH_REGION. Override any voice with AZURE_VOICE_<LANG>, e.g. AZURE_VOICE_EN.
# Yoruba/Hausa/Igbo have no Azure neural voice that I could confirm -> the UI falls back to the browser voice for those.
AZURE_KEY, AZURE_REGION = os.getenv("AZURE_SPEECH_KEY"), os.getenv("AZURE_SPEECH_REGION")
VOICES = {"en": ("en-GB-RyanNeural", "en-GB"), "fr": ("fr-FR-HenriNeural", "fr-FR"),
          "pt": ("pt-PT-DuarteNeural", "pt-PT"), "pcm": ("en-NG-AbeoNeural", "en-NG")}
PERSONAS = {
    "analyst": "a tactical analyst: dense, precise, cite the numbers (xG, momentum index, pressure, speeds).",
    "casual": "an excited, friendly commentator talking to a casual fan: plain words, no jargon, vivid and short.",
    "player_focus": "a commentator following ONE player: centre every line on that player's involvement and milestones.",
}
TEAMS = {
    "HOME": {"name": "Arsenal", "strength": 1.12, "players": [
        "David Raya", "Ben White", "William Saliba", "Gabriel Magalhães", "Riccardo Calafiori", "Declan Rice",
        "Martin Ødegaard", "Martin Zubimendi", "Bukayo Saka", "Viktor Gyökeres", "Gabriel Martinelli"]},
    "AWAY": {"name": "Chelsea", "strength": 0.98, "players": [
        "Robert Sánchez", "Reece James", "Levi Colwill", "Trevoh Chalobah", "Marc Cucurella", "Moisés Caicedo",
        "Enzo Fernández", "Cole Palmer", "Pedro Neto", "João Pedro", "Estêvão"]},
}
OPP = {"HOME": "AWAY", "AWAY": "HOME"}


# ----------------------------------------------------------------------------------------------
# (a) INGEST - seedable synthetic match generator (pitch 0-100 x 0-100, HOME attacks toward x=100)
# ----------------------------------------------------------------------------------------------
def generate_match(seed: int = 7):
    rng = random.Random(seed)
    events, base_km = [], {p: rng.uniform(8.5, 12.0) for k in TEAMS for p in TEAMS[k]["players"]}
    t, poss, bx, by = 0, "HOME", 50.0, 50.0
    ab = lambda k, x: x if k == "HOME" else 100 - x  # team frame -> absolute pitch

    def pick(k, x):
        lo, hi = (0, 5) if x < 30 else (3, 9) if x < 65 else (6, 11)
        return TEAMS[k]["players"][rng.randrange(lo, hi)]

    def add(k, player, typ, x, y, **kw):
        e = dict(id=len(events), t=t, minute=t // 60, team=k, player=player, type=typ,
                 x=round(ab(k, x), 1), y=round(y, 1), end_x=None, end_y=None, success=None,
                 distance_m=None, ball_speed_kmh=None, xg=None, outcome=None, pressured=False,
                 player_speed_kmh=round(rng.uniform(31, 36) if rng.random() < 0.012 else rng.uniform(7, 26), 1),
                 player_distance_km=round(base_km[player] * t / 5400, 2))
        e.update(kw)
        events.append(e)

    while True:
        t += rng.randint(3, 8)
        if t >= 5400:
            break
        d = OPP[poss]
        phase = 0.12 * math.sin(t / 5400 * math.pi * 5)  # dominance swings through the match
        s = TEAMS[poss]["strength"] + (phase if poss == "HOME" else -phase)
        pressured = rng.random() < 0.22
        if pressured:
            add(d, pick(d, 100 - bx), "pressure", 100 - bx, 100 - by)
        r = rng.random()
        if bx > 72 and r < 0.22:  # shot
            gd = math.hypot(100 - bx, by - 50)
            xg = round(min(0.65, 0.7 * math.exp(-gd / 14) * (0.9 + 0.1 * s)), 2)
            goal = rng.random() < xg * 0.55
            out = "goal" if goal else rng.choices(["saved", "off_target", "blocked"], [5, 4, 3])[0]
            add(poss, pick(poss, bx), "shot", bx, by, xg=xg, outcome=out,
                ball_speed_kmh=round(rng.uniform(65, 118), 1), pressured=pressured)
            if goal:
                poss, bx, by = d, 50.0, 50.0
                add(poss, pick(poss, 50), "possession_change", 50, 50, outcome="kickoff")
            elif out == "blocked" and rng.random() < 0.5:
                bx, by = 88.0, rng.uniform(30, 70)
            else:
                poss, bx, by = d, rng.uniform(6, 20), rng.uniform(35, 65)
                add(poss, pick(poss, bx), "possession_change", bx, by,
                    outcome="goal_kick" if out == "off_target" else "keeper_claim")
        elif r < 0.09 + (0.07 if pressured else 0):  # tackle
            tk = pick(d, 100 - bx)
            add(d, tk, "tackle", 100 - bx, 100 - by, success=True)
            poss, bx, by = d, 100 - bx, 100 - by
            add(poss, tk, "possession_change", bx, by, outcome="tackle_won")
        else:  # pass
            nx = min(98, max(2, bx + rng.gauss(7, 9)))
            ny = min(97, max(3, by + rng.gauss(0, 12)))
            ln = math.hypot(nx - bx, ny - by) * 1.05
            p_ok = min(0.97, max(0.35, 0.94 - 0.006 * ln - (0.08 if pressured else 0) + 0.04 * (s - 1)))
            ok = rng.random() < p_ok
            add(poss, pick(poss, bx), "pass", bx, by, end_x=round(ab(poss, nx), 1), end_y=round(ny, 1),
                success=ok, distance_m=round(ln, 1), pressured=pressured,
                ball_speed_kmh=round(25 + ln * 1.3 + rng.gauss(0, 5), 1))
            if ok:
                bx, by = nx, ny
            else:
                poss, bx, by = d, 100 - nx, 100 - ny
                add(poss, pick(poss, bx), "possession_change", bx, by, outcome="turnover")
    return events


# ----------------------------------------------------------------------------------------------
# (b) INTERPRET - stats, pass difficulty, momentum & control-vs-chaos
# ----------------------------------------------------------------------------------------------
def tf(e, key):
    """Coordinate in the acting team's attacking frame (own goal = 0, opponent goal = 100)."""
    v = e[key]
    return v if e["team"] == "HOME" else 100 - v


def pass_difficulty(e):
    """0-100 rating from length, forward progress, destination (final third) and pressure."""
    gain = max(tf(e, "end_x") - tf(e, "x"), 0)
    return round(min(100, 12 + 1.1 * e["distance_m"] + 0.9 * gain
                     + (18 if tf(e, "end_x") >= 67 else 0) + (15 if e["pressured"] else 0)), 1)


def compute_state(evs):
    out = {"score": {"HOME": 0, "AWAY": 0}}
    for k in TEAMS:
        mine = [e for e in evs if e["team"] == k]
        theirs = [e for e in evs if e["team"] == OPP[k]]
        passes = [e for e in mine if e["type"] == "pass"]
        ok = [e for e in passes if e["success"]]
        shots = [e for e in mine if e["type"] == "shot"]
        opp_passes = len([e for e in theirs if e["type"] == "pass"])
        defs = len([e for e in mine if e["type"] in ("tackle", "pressure")])
        atk = [e for e in evs if (e["x"] >= 50 if k == "HOME" else e["x"] < 50)]
        diffs = [pass_difficulty(e) for e in passes]
        out["score"][k] = len([e for e in shots if e["outcome"] == "goal"])
        out[k] = dict(
            name=TEAMS[k]["name"],
            possession_pct=round(100 * len(passes) / max(len(passes) + opp_passes, 1), 1),
            territory_pct=round(100 * len(atk) / max(len(evs), 1), 1),
            passes=len(passes), pass_accuracy_pct=round(100 * len(ok) / max(len(passes), 1), 1),
            avg_pass_difficulty=round(sum(diffs) / max(len(diffs), 1), 1),
            shots=len(shots), shots_on_target=len([e for e in shots if e["outcome"] in ("goal", "saved")]),
            xg=round(sum(e["xg"] for e in shots), 2),
            tackles=len([e for e in mine if e["type"] == "tackle"]),
            pressures=len([e for e in mine if e["type"] == "pressure"]),
            ppda=round(opp_passes / max(defs, 1), 1),
            top_player_speed_kmh=max([e["player_speed_kmh"] for e in mine] or [0]),
            max_shot_speed_kmh=max([e["ball_speed_kmh"] for e in shots] or [0]),
            max_pass_speed_kmh=max([e["ball_speed_kmh"] for e in passes] or [0]))
    return out


def build_momentum(events):
    """Rolling 5-minute index in [-100, 100] (+ = HOME control) plus a control-vs-chaos label."""
    series = []
    for m in range(1, 91):
        w = [e for e in events if m * 60 - 300 < e["t"] <= m * 60]

        def share(f):
            a = sum(f(e) for e in w if e["team"] == "HOME")
            b = sum(f(e) for e in w if e["team"] == "AWAY")
            return 0.0 if a + b == 0 else (a - b) / (a + b)
        h = len([e for e in w if e["x"] >= 50]); a_ = len([e for e in w if e["x"] < 50])
        terr = 0.0 if h + a_ == 0 else (h - a_) / (h + a_)
        idx = 100 * (0.35 * share(lambda e: e["type"] == "pass") + 0.25 * terr
                     + 0.25 * share(lambda e: e["xg"] or 0) + 0.15 * share(lambda e: e["type"] == "pressure"))
        turns = len([e for e in w if e["outcome"] in ("turnover", "tackle_won")]) / 5
        state = "controlled" if turns < 2.0 else "balanced" if turns < 2.8 else "chaotic"
        series.append(dict(minute=m, index=round(idx, 1), turnovers_per_min=round(turns, 2), state=state))
    return series


# ----------------------------------------------------------------------------------------------
# (c) EXPLAIN - key moments, each with evidence-backed "why it matters"
# ----------------------------------------------------------------------------------------------
def detect_moments(events, mom):
    M, nm = [], lambda k: TEAMS[k]["name"]
    score, xg_run, seen = {"HOME": 0, "AWAY": 0}, {"HOME": 0.0, "AWAY": 0.0}, set()
    n_key, last_press = 0, -9999
    tm = lambda k, mi: mi["index"] if k == "HOME" else -mi["index"]

    def add(t, typ, pr, team, player, headline, why, simple, facts):
        M.append(dict(t=t, minute=t // 60, type=typ, priority=pr, team=team, player=player,
                      headline=headline, why=why, simple=simple, facts=facts))

    add(0, "kickoff", 1, None, None, "Kick-off", f"{nm('HOME')} vs {nm('AWAY')} begins.",
        f"We're underway: {nm('HOME')} vs {nm('AWAY')}!", {})
    for i, e in enumerate(events):
        k, o, mn = e["team"], OPP[e["team"]], e["minute"]
        mi = mom[min(max(mn, 1), 90) - 1]
        if e["type"] == "shot":
            xg_run[k] += e["xg"]
            if e["outcome"] == "goal":
                score[k] += 1
                kind = "a clinical finish from a low-probability chance" if e["xg"] < 0.15 else "the reward for a high-quality chance"
                add(e["t"], "goal", 3, k, e["player"], f"GOAL - {e['player']} ({nm(k)})",
                    f"{e['player']} scores in minute {mn}: {score['HOME']}-{score['AWAY']}. The chance was worth {e['xg']:.2f} xG, "
                    f"{kind}. {nm(k)} xG so far {xg_run[k]:.2f} vs {xg_run[o]:.2f}. Pre-goal momentum for {nm(k)}: {tm(k, mi):+.0f} ({mi['state']}). "
                    f"Shot speed {e['ball_speed_kmh']:.0f} km/h.",
                    f"GOAL! {e['player']} scores for {nm(k)}! It's {score['HOME']}-{score['AWAY']}.",
                    dict(xg=e["xg"], shot_speed_kmh=e["ball_speed_kmh"], score=dict(score), momentum=tm(k, mi)))
            elif e["xg"] >= 0.25:
                add(e["t"], "big_chance", 2, k, e["player"], f"Big chance - {e['player']}",
                    f"{e['player']}'s effort ({e['outcome'].replace('_', ' ')}) was a {e['xg']:.2f} xG chance: chances like this are scored about "
                    f"{e['xg'] * 100:.0f}% of the time. {nm(k)} now have {xg_run[k]:.2f} xG.",
                    f"So close! {e['player']} had a great chance but it went {e['outcome'].replace('_', ' ')}.",
                    dict(xg=e["xg"], outcome=e["outcome"]))
        elif e["type"] == "pressure":
            last_press = max(last_press, e["t"]) if e["team"] else last_press
        elif e["type"] == "tackle" and tf(e, "x") >= 60:
            prev = [x for x in events[max(0, i - 4):i] if x["type"] == "pressure" and x["team"] == k and e["t"] - x["t"] <= 10]
            nxt = [x for x in events[i + 1:i + 12] if x["type"] == "shot" and x["team"] == k and x["t"] - e["t"] <= 20]
            if prev and (nxt or e["t"] - last_press >= 600 or ("trap", mn // 10) not in seen):
                seen.add(("trap", mn // 10))
                led = f" It led to a {nxt[0]['xg']:.2f} xG shot within {nxt[0]['t'] - e['t']}s." if nxt else ""
                add(e["t"], "press_win", 2 if nxt else 1, k, e["player"], f"Pressing trap - {e['player']}",
                    f"{nm(k)} win the ball {tf(e, 'x'):.0f}% up the pitch, {e['t'] - prev[0]['t']}s after a pressure event: "
                    f"a coordinated press, not luck.{led}",
                    f"{e['player']} wins the ball high up the pitch after a great team press!{' A shot follows.' if nxt else ''}",
                    dict(tackle_x=tf(e, "x"), led_to_shot=bool(nxt)))
        if e["player_speed_kmh"] >= 33 and ("sprint", e["player"]) not in seen:
            seen.add(("sprint", e["player"]))
            add(e["t"], "sprint", 1, k, e["player"], f"Sprint alert - {e['player']}",
                f"{e['player']} reaches {e['player_speed_kmh']:.1f} km/h, above the 33 km/h alert threshold - usually a transition or recovery run.",
                f"{e['player']} is flying - {e['player_speed_kmh']:.0f} km/h!", dict(speed_kmh=e["player_speed_kmh"]))
        for km in (5, 10):
            if e["player_distance_km"] >= km and ("km", km, e["player"]) not in seen:
                seen.add(("km", km, e["player"]))
                add(e["t"], "milestone", 1, k, e["player"], f"{e['player']}: {km} km covered",
                    f"{e['player']} passes {km} km covered in minute {mn} ({e['player_distance_km']:.1f} km).",
                    f"{e['player']} has now run {km} km!", dict(distance_km=e["player_distance_km"]))
        if e["type"] == "pass" and e["success"] and n_key < 8:
            df = pass_difficulty(e)
            if df >= 85:
                n_key += 1
                add(e["t"], "key_pass", 2, k, e["player"], f"Difficult pass - {e['player']}",
                    f"{e['player']} completes a {e['distance_m']:.0f} m pass rated {df:.0f}/100 difficulty"
                    f"{' under pressure' if e['pressured'] else ''}, travelling at {e['ball_speed_kmh']:.0f} km/h.",
                    f"What a ball from {e['player']}!", dict(difficulty=df, distance_m=e["distance_m"]))
    last = -99  # momentum shifts
    for m in range(6, 91):
        sh = mom[m - 1]["index"] - mom[m - 6]["index"]
        if abs(sh) >= 30 and m - last >= 8:
            last, k = m, "HOME" if sh > 0 else "AWAY"
            a, b = tm(k, mom[m - 6]), tm(k, mom[m - 1])
            add(m * 60, "momentum_shift", 2, k, None, f"Momentum swings to {nm(k)}",
                f"Control is shifting to {nm(k)}: momentum index {a:+.0f} -> {b:+.0f} in 5 minutes "
                f"(possession, territory, xG and pressing blend). Game state: {mom[m - 1]['state']} "
                f"({mom[m - 1]['turnovers_per_min']} turnovers/min).",
                f"{nm(k)} are taking over the game right now.", dict(from_=a, to=b, state=mom[m - 1]["state"]))
    for t_, typ, label in ((2700, "half_time", "Half-time"), (5400, "full_time", "Full-time")):
        st = compute_state([e for e in events if e["t"] <= t_])
        sc, h, a = st["score"], st["HOME"], st["AWAY"]
        txt = (f"{label}: {nm('HOME')} {sc['HOME']}-{sc['AWAY']} {nm('AWAY')}. xG {h['xg']}-{a['xg']}, possession "
               f"{h['possession_pct']}%-{a['possession_pct']}%, shots {h['shots']}-{a['shots']}.")
        add(t_, typ, 3, None, None, f"{label}: {sc['HOME']}-{sc['AWAY']}", txt,
            f"{label}! {nm('HOME')} {sc['HOME']}-{sc['AWAY']} {nm('AWAY')}.", dict(score=sc))
    M.sort(key=lambda m: m["t"])
    for n, m in enumerate(M):
        m["id"] = f"m{n}"
    return M


# ----------------------------------------------------------------------------------------------
# (e) PERSONALIZE - Claude, constrained to computed facts
# ----------------------------------------------------------------------------------------------
def ask_claude(system, user, max_tokens=500):
    if CLIENT is None:
        return None
    try:
        r = CLIENT.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                                   messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text").strip()
    except Exception as ex:
        print("Claude error:", ex)
        return None


def style_prompt(persona, lang, fav, focus):
    s = (f"You are {PERSONAS.get(persona, PERSONAS['casual'])} Write in {LANGS.get(lang, 'English')}. "
         "Use ONLY the facts in the JSON provided; never invent players, numbers or events. Keep names, numbers and units exact. "
         "Explain WHY the moment matters, not just that it happened.")
    if fav in TEAMS:
        s += f" The viewer supports {TEAMS[fav]['name']}: frame events from their perspective (still honest and fair)."
    if focus:
        s += f" The viewer is following {focus}; mention their involvement when the facts allow."
    return s


EVENTS = generate_match()
MOM = build_momentum(EVENTS)
MOMENTS = detect_moments(EVENTS, MOM)
CACHE = {}
LAYOUT = {"goal": "full_banner", "half_time": "full_banner", "full_time": "full_banner", "kickoff": "full_banner",
          "big_chance": "lower_third", "press_win": "lower_third", "momentum_shift": "momentum_bar",
          "sprint": "player_chip", "milestone": "player_chip", "key_pass": "player_chip", "player_pulse": "player_chip"}


def player_summary(upto_s, p):
    mine = [e for e in EVENTS if e["player"] == p and e["t"] <= upto_s]
    if not mine:
        return None
    ps = [e for e in mine if e["type"] == "pass"]
    ok = len([e for e in ps if e["success"]])
    return dict(passes=len(ps), acc=round(100 * ok / max(len(ps), 1)), shots=len([e for e in mine if e["type"] == "shot"]),
                tackles=len([e for e in mine if e["type"] == "tackle"]), km=mine[-1]["player_distance_km"],
                top=max(e["player_speed_kmh"] for e in mine), team=mine[0]["team"])


def player_alerts(p, upto_s, from_s=0, speed_kmh=30.0, step_km=2.0, diff=80.0):
    """Threshold alerts for one player: sprint speed, distance milestones, shots, difficult passes.
    (Goals, big chances and pass difficulty >= 85 are already covered by key moments, so they are not repeated here.)"""
    step_km, cards, last_speed, last_step = max(step_km, 0.5), [], -9999, 0

    def add(e, kind, pr, headline, body, **data):
        if e["t"] >= from_s:
            cards.append(dict(card_id=f"alert-{kind}-{e['id']}-{p}", show_at_s=e["t"], duration_s={3: 12, 2: 8, 1: 5}[pr],
                              layout="player_chip", type="player_alert", priority=pr, team=e["team"], player=p,
                              headline=headline, body=body, tone="alert", data=dict(kind=kind, **data)))
    for e in EVENTS:
        if e["player"] != p or e["t"] > upto_s:
            continue
        mn = e["minute"]
        if (e["player_speed_kmh"] or 0) >= speed_kmh and e["t"] - last_speed >= 120:
            last_speed = e["t"]
            add(e, "speed", 2, f"⚡ {p}: {e['player_speed_kmh']:.1f} km/h",
                f"Sprint alert at {mn}': {p} passes the {speed_kmh:.0f} km/h threshold ({e['player_speed_kmh']:.1f} km/h).",
                speed_kmh=e["player_speed_kmh"], threshold=speed_kmh)
        while (e["player_distance_km"] or 0) >= (last_step + 1) * step_km:
            last_step += 1
            add(e, "distance", 1, f"🏃 {p}: {last_step * step_km:.1f} km covered",
                f"{p} has covered {e['player_distance_km']:.1f} km after {mn}' (alert every {step_km:g} km).",
                distance_km=e["player_distance_km"], threshold=last_step * step_km)
        if e["type"] == "shot" and e["outcome"] != "goal" and (e["xg"] or 0) < 0.25:
            add(e, "shot", 1, f"🎯 {p}: shot ({e['outcome'].replace('_', ' ')})",
                f"{p} shoots at {mn}': {e['outcome'].replace('_', ' ')}, {e['xg']:.2f} xG, {e['ball_speed_kmh']:.0f} km/h.",
                xg=e["xg"], outcome=e["outcome"])
        if e["type"] == "pass" and e["success"] and e["end_x"] is not None:
            df = pass_difficulty(e)
            if diff <= df < 85:
                add(e, "pass", 1, f"🎩 {p}: difficult pass ({df:.0f}/100)",
                    f"{p} completes a {e['distance_m']:.0f} m pass rated {df:.0f}/100{' under pressure' if e['pressured'] else ''}.",
                    difficulty=df, threshold=diff)
    return cards


def build_cards(upto_s, persona="casual", fav=None, focus=None, from_s=0, speed_kmh=30.0, step_km=2.0):
    cards = []
    for m in MOMENTS:
        if not (from_s <= m["t"] <= upto_s):
            continue
        pr = min(3, m["priority"] + (1 if fav and m["team"] == fav else 0))
        if persona == "player_focus" and focus and m["player"] != focus and m["type"] not in ("goal", "half_time", "full_time"):
            continue
        if persona == "casual" and pr < 2:
            continue
        if persona == "player_focus" and focus and m["type"] in ("sprint", "milestone"):
            continue  # replaced by the threshold alerts below
        cards.append(dict(card_id=f"{m['id']}-{persona}", show_at_s=m["t"], duration_s={3: 12, 2: 8, 1: 5}[pr],
                          layout=LAYOUT.get(m["type"], "lower_third"), type=m["type"], priority=pr, team=m["team"],
                          player=m["player"], headline=m["headline"], body=m["simple"] if persona == "casual" else m["why"],
                          tone="favorite" if fav and m["team"] == fav else "neutral",
                          data=m["facts"] if persona != "casual" else {}))
    if persona == "player_focus" and focus:
        cards += player_alerts(focus, upto_s, from_s, speed_kmh, step_km)
        for mn in range(10, 91, 10):
            ps = player_summary(mn * 60, focus)
            if ps and from_s <= mn * 60 <= upto_s:
                cards.append(dict(card_id=f"pulse{mn}-{focus}", show_at_s=mn * 60, duration_s=8, layout="player_chip", type="player_pulse",
                                  priority=2, team=ps["team"], player=focus, headline=f"{focus} - minute {mn}",
                                  body=f"{ps['passes']} passes ({ps['acc']}% accurate), {ps['shots']} shots, {ps['tackles']} tackles, "
                                       f"{ps['km']} km covered, top speed {ps['top']} km/h.",
                                  tone="neutral", data=ps))
    return sorted(cards, key=lambda c: c["show_at_s"])


def translate_cards(cards, lang):
    if lang == "en" or lang not in LANGS:
        return cards
    need = [c for c in cards if (c["card_id"], lang) not in CACHE]
    if need and CLIENT:
        out = ask_claude(f"Translate football overlay text into {LANGS[lang]}. Keep names, numbers and units unchanged. "
                         "Return ONLY a JSON array of objects with keys id, headline, body.",
                         json.dumps([dict(id=c["card_id"], headline=c["headline"], body=c["body"]) for c in need], ensure_ascii=False), 3500)
        try:
            for o in json.loads(out.replace("```json", "").replace("```", "").strip()):
                CACHE[(o["id"], lang)] = o
        except Exception:
            pass
    for c in cards:
        o = CACHE.get((c["card_id"], lang))
        if o:
            c["headline"], c["body"] = o["headline"], o["body"]
    return cards


# ----------------------------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------------------------
app = FastAPI(title="PitchLens API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NarrateReq(BaseModel):
    moment_id: str
    persona: str = "casual"
    language: str = "en"
    favorite_team: Optional[str] = None
    focus_player: Optional[str] = None


class RecapReq(BaseModel):
    minute: int = 90
    persona: str = "casual"
    language: str = "en"
    favorite_team: Optional[str] = None
    focus_player: Optional[str] = None


class Detection(BaseModel):  # shape a video model would emit (auto-eventing hook)
    t: int
    type: str
    team: str
    player: str
    x: float
    y: float
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    success: Optional[bool] = None
    distance_m: Optional[float] = None
    ball_speed_kmh: Optional[float] = None
    xg: Optional[float] = None
    outcome: Optional[str] = None


class Pt(BaseModel):
    x: float
    y: float


class Holder(BaseModel):
    player: str
    team: str  # "HOME" | "AWAY"


class PlayerPos(BaseModel):
    player: str
    team: str
    x: float
    y: float


class Frame(BaseModel):
    """One tracked frame from a video model. Coordinates: pitch units 0-100 (HOME attacks toward x=100); 100 units ~ 105 m."""
    t: float                              # seconds since kick-off
    ball: Optional[Pt] = None
    holder: Optional[Holder] = None       # player currently in possession (ball-to-player association)
    players: List[PlayerPos] = []         # optional player positions (for speed)


class TrackingFeed(BaseModel):
    frames: List[Frame]
    commit: bool = False                  # True = append derived events to the live match and re-run the pipeline
    speed_threshold_kmh: float = 30.0
    min_pass_m: float = 3.0               # ignore holder flicker when the ball barely moved
    max_pass_s: float = 2.5               # longest plausible ball flight between two holders


class VoiceReq(BaseModel):
    text: str
    language: str = "en"
    rate: str = "+0%"


@app.get("/health")
def health():
    return {"ok": True, "claude_enabled": CLIENT is not None, "model": MODEL, "events": len(EVENTS), "moments": len(MOMENTS),
            "azure_speech_enabled": bool(AZURE_KEY and AZURE_REGION), "azure_voice_langs": list(VOICES)}


@app.get("/match/meta")
def match_meta():
    return {"match_id": "SYN-0001", "duration_min": 90, "teams": {k: v["name"] for k, v in TEAMS.items()},
            "languages": LANGS, "personas": list(PERSONAS)}


@app.get("/players")
def players():
    return [{"name": p, "team": k, "team_name": TEAMS[k]["name"]} for k in TEAMS for p in TEAMS[k]["players"]]


@app.get("/events")
def events(from_s: int = 0, to_s: int = 5400, limit: int = 200):
    """Raw ingest feed (what a live feed would push)."""
    return [e for e in EVENTS if from_s <= e["t"] <= to_s][:limit]


@app.get("/match/state")
def match_state(minute: int = Query(90, ge=0, le=90)):
    return compute_state([e for e in EVENTS if e["t"] <= minute * 60])


@app.get("/match/momentum")
def match_momentum():
    return MOM


@app.get("/match/moments")
def match_moments(minute: int = Query(90, ge=0, le=90)):
    return [m for m in MOMENTS if m["t"] <= minute * 60]


@app.get("/overlays")
def overlays(minute: int = Query(90, ge=0, le=90), persona: str = "casual", language: str = "en",
             favorite_team: Optional[str] = None, focus_player: Optional[str] = None, last: int = 8,
             speed_kmh: float = Query(30.0, ge=20, le=40), distance_step_km: float = Query(2.0, ge=0.5, le=5)):
    """Timed, machine-readable overlay cards (show_at_s / duration_s / layout) for a graphics layer."""
    cards = build_cards(minute * 60, persona, favorite_team, focus_player, 0, speed_kmh, distance_step_km)[-last:]
    return translate_cards(cards, language)


@app.get("/stream")
async def stream(speed: float = 60.0, persona: str = "casual", language: str = "en",
                 favorite_team: Optional[str] = None, focus_player: Optional[str] = None,
                 speed_kmh: float = 30.0, distance_step_km: float = 2.0):
    """Server-Sent Events: overlay cards released in (accelerated) real time. speed = match-seconds per real second."""
    cards = build_cards(5400, persona, favorite_team, focus_player, 0, speed_kmh, distance_step_km)

    async def gen():
        prev = 0
        for c in cards:
            await asyncio.sleep(min(max((c["show_at_s"] - prev) / speed, 0), 3))
            prev = c["show_at_s"]
            yield f"data: {json.dumps(translate_cards([c], language)[0], ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/narrate")
def narrate(req: NarrateReq):
    m = next((x for x in MOMENTS if x["id"] == req.moment_id), None)
    if not m:
        raise HTTPException(404, "moment not found")
    key = (req.moment_id, req.persona, req.language, req.favorite_team, req.focus_player)
    if key in CACHE:
        return CACHE[key]
    st = compute_state([e for e in EVENTS if e["t"] <= m["t"]])
    facts = dict(moment=m, score=st["score"], teams={k: st[k] for k in TEAMS}, momentum=MOM[min(max(m["minute"], 1), 90) - 1])
    txt = ask_claude(style_prompt(req.persona, req.language, req.favorite_team, req.focus_player) +
                     " Write 1-3 sentences of live commentary for this moment.", json.dumps(facts, ensure_ascii=False), 300)
    res = {"text": txt or (m["simple"] if req.persona == "casual" else m["why"]), "engine": "claude" if txt else "template"}
    CACHE[key] = res
    return res


@app.post("/recap")
def recap(req: RecapReq):
    upto = req.minute * 60
    evs = [e for e in EVENTS if e["t"] <= upto]
    st = compute_state(evs)
    top = [m for m in MOMENTS if m["t"] <= upto and m["priority"] >= 2][:16]
    impact = {}
    for m in top:
        if m["player"]:
            impact[m["player"]] = impact.get(m["player"], 0) + m["priority"]
    facts = dict(minute=req.minute, score=st["score"], teams={k: st[k] for k in TEAMS}, key_moments=[
        dict(minute=m["minute"], type=m["type"], team=m["team"], player=m["player"], why=m["why"]) for m in top],
        momentum_every_15min=[MOM[i] for i in (14, 29, 44, 59, 74, 89) if i < req.minute],
        standout_player=max(impact, key=impact.get) if impact else None)
    txt = ask_claude(style_prompt(req.persona, req.language, req.favorite_team, req.focus_player) +
                     " Write an automated match recap (120-180 words): result, the story of the match, turning points and why they mattered, standout player.",
                     json.dumps(facts, ensure_ascii=False), 900)
    if not txt:
        sc = st["score"]
        txt = (f"{TEAMS['HOME']['name']} {sc['HOME']}-{sc['AWAY']} {TEAMS['AWAY']['name']} after {req.minute}'. "
               + " ".join(m["simple"] if req.persona == "casual" else m["why"] for m in top[:5]))
    return {"recap": txt, "engine": "claude" if CLIENT and txt else "template", "facts": facts}


def commit_events(new):
    """Append events, re-index, and re-run interpret + explain."""
    global EVENTS, MOM, MOMENTS
    EVENTS = sorted(EVENTS + new, key=lambda e: e["t"])
    for i, e in enumerate(EVENTS):
        e["id"] = i
    MOM = build_momentum(EVENTS)
    MOMENTS = detect_moments(EVENTS, MOM)
    CACHE.clear()


def tracking_to_events(frames, speed_kmh=30.0, min_pass_m=3.0, max_pass_s=2.5):
    """AUTO-EVENTING HOOK. Turns video-model output (ball track + possession holder + optional player positions)
    into PitchLens events with the pass and speed metrics, with no hand tagging:
      - holder changes while the ball moved >= min_pass_m within max_pass_s -> `pass` (success if same team, else turnover)
        with distance_m and ball_speed_kmh = distance / flight time
      - player position deltas -> `run` events (player_speed_kmh) when speed >= speed_kmh (10 s cool-down per player)
    Plug a real detector (e.g. YOLO + ByteTrack + pitch homography + ball-possession association) in front of this;
    it only has to emit `Frame` objects. No model is shipped here."""
    out, last, pos, last_run = [], None, {}, {}

    def ev(t, team, player, typ, x, y, **kw):
        e = dict(id=0, t=int(round(t)), minute=int(t) // 60, team=team, player=player, type=typ, x=round(x, 1), y=round(y, 1),
                 end_x=None, end_y=None, success=None, distance_m=None, ball_speed_kmh=None, xg=None, outcome=None,
                 pressured=False, player_speed_kmh=0.0, player_distance_km=0.0)
        e.update(kw)
        out.append(e)
    for f in sorted(frames, key=lambda f: f.t):
        for p in f.players:
            if p.player in pos:
                t0, x0, y0 = pos[p.player]
                dt = f.t - t0
                if 0.2 <= dt <= 5:
                    v = math.hypot(p.x - x0, p.y - y0) * 1.05 / dt * 3.6
                    if speed_kmh <= v <= 45 and f.t - last_run.get(p.player, -999) >= 10:
                        last_run[p.player] = f.t
                        ev(f.t, p.team, p.player, "run", p.x, p.y, player_speed_kmh=round(v, 1))
            pos[p.player] = (f.t, p.x, p.y)
        if f.holder and f.ball:
            if last and f.holder.player != last["player"]:
                dt, d = f.t - last["t"], math.hypot(f.ball.x - last["x"], f.ball.y - last["y"]) * 1.05
                if 0 < dt <= max_pass_s and d >= min_pass_m:
                    ok = f.holder.team == last["team"]
                    ev(last["t"], last["team"], last["player"], "pass", last["x"], last["y"], end_x=round(f.ball.x, 1),
                       end_y=round(f.ball.y, 1), success=ok, distance_m=round(d, 1), ball_speed_kmh=round(min(d / dt * 3.6, 150), 1))
                    if not ok:
                        ev(f.t, f.holder.team, f.holder.player, "possession_change", f.ball.x, f.ball.y, outcome="turnover")
            last = dict(player=f.holder.player, team=f.holder.team, t=f.t, x=f.ball.x, y=f.ball.y)
    return out


@app.post("/autoevent/tracking")
def autoevent_tracking(feed: TrackingFeed):
    """Video-model output in -> pass/speed events out (optionally committed to the live match)."""
    det = tracking_to_events(feed.frames, feed.speed_threshold_kmh, feed.min_pass_m, feed.max_pass_s)
    if feed.commit and det:
        commit_events(det)
    return {"frames": len(feed.frames), "events": det, "committed": bool(feed.commit and det), "total_events": len(EVENTS)}


@app.post("/autoevent/simulate")
def autoevent_simulate(from_s: int = Query(0, ge=0, le=5400), to_s: int = Query(900, ge=0, le=5400), noise: float = Query(0.4, ge=0, le=3),
                       seed: int = 1, include_frames: bool = False):
    """Demo of the hook WITHOUT a model: fabricates noisy 'video model' tracking frames from the synthetic ground truth,
    runs them through tracking_to_events(), and scores the detections against the truth. Nothing is committed."""
    rng, evs, frames = random.Random(seed), [e for e in EVENTS if from_s <= e["t"] <= to_s], []
    jit = lambda: rng.gauss(0, noise)
    for i, e in enumerate(evs):
        if e["type"] == "pass" and e["end_x"] is not None and e["team"] in TEAMS and e["player"] in TEAMS[e["team"]]["players"]:
            mates, foes = TEAMS[e["team"]]["players"], TEAMS[OPP[e["team"]]]["players"]
            nxt = next((x for x in evs[i + 1:i + 4] if x["type"] in ("pass", "possession_change")), None)
            if e["success"]:
                rec = nxt["player"] if nxt and nxt["team"] == e["team"] and nxt["player"] != e["player"] else mates[(mates.index(e["player"]) + 1) % 11]
                rteam = e["team"]
            else:
                rec, rteam = (nxt["player"] if nxt and nxt["team"] != e["team"] else foes[0]), OPP[e["team"]]
            travel = max(0.4, e["distance_m"] / max(e["ball_speed_kmh"] / 3.6, 1))
            frames.append(Frame(t=e["t"], ball=Pt(x=e["x"] + jit(), y=e["y"] + jit()), holder=Holder(player=e["player"], team=e["team"])))
            frames.append(Frame(t=e["t"] + travel, ball=Pt(x=e["end_x"] + jit(), y=e["end_y"] + jit()), holder=Holder(player=rec, team=rteam)))
        if (e["player_speed_kmh"] or 0) >= 30:
            step = e["player_speed_kmh"] / 3.6 / 1.05  # pitch units covered in 1 s
            frames.append(Frame(t=e["t"] - 1, players=[PlayerPos(player=e["player"], team=e["team"], x=e["x"] + jit(), y=e["y"] + jit())]))
            frames.append(Frame(t=e["t"], players=[PlayerPos(player=e["player"], team=e["team"], x=e["x"] + step + jit(), y=e["y"] + jit())]))
    det = tracking_to_events(frames)
    gt_p, dp = [e for e in evs if e["type"] == "pass"], [d for d in det if d["type"] == "pass"]
    gt_s, ds = [e for e in evs if (e["player_speed_kmh"] or 0) >= 30], [d for d in det if d["type"] == "run"]
    used, errs = set(), []
    for g in gt_p:
        j = next((j for j, d in enumerate(dp) if j not in used and d["player"] == g["player"] and abs(d["t"] - g["t"]) <= 1), None)
        if j is not None:
            used.add(j)
            errs.append(abs(dp[j]["ball_speed_kmh"] - g["ball_speed_kmh"]))
    s_hit = len([g for g in gt_s if any(d["player"] == g["player"] and abs(d["t"] - g["t"]) <= 2 for d in ds)])
    res = {"window_s": [from_s, to_s], "frames": len(frames), "noise_units": noise,
           "ground_truth": {"passes": len(gt_p), "sprints_ge_30kmh": len(gt_s)},
           "detected": {"passes": len(dp), "runs": len(ds)},
           "pass_recall": round(len(errs) / max(len(gt_p), 1), 3), "pass_precision": round(len(errs) / max(len(dp), 1), 3),
           "ball_speed_mae_kmh": round(sum(errs) / max(len(errs), 1), 2), "sprint_recall": round(s_hit / max(len(gt_s), 1), 3),
           "sample_events": [d for d in det if d["type"] in ("pass", "run")][:8],
           "note": "Frames are simulated from ground truth + noise. Swap in a real detector to produce Frame objects."}
    if include_frames:
        res["frames_payload"] = {"frames": [f.model_dump() for f in frames[:400]]}
    return res


@app.post("/autoevent/ingest")
def autoevent_ingest(dets: List[Detection]):
    """Already-structured detections (e.g. from a tagging tool) in; the pipeline re-interprets everything."""
    new = []
    for d in dets:
        e = dict(id=0, minute=d.t // 60, pressured=False, player_speed_kmh=0.0, player_distance_km=0.0, **d.model_dump())
        if e["type"] == "pass":
            if e["end_x"] is None or e["end_y"] is None:
                raise HTTPException(422, "pass detections need end_x and end_y")
            e["distance_m"] = e["distance_m"] or round(math.hypot(e["end_x"] - e["x"], e["end_y"] - e["y"]) * 1.05, 1)
            e["ball_speed_kmh"] = e["ball_speed_kmh"] or 0.0
        new.append(e)
    commit_events(new)
    return {"ingested": len(dets), "total_events": len(EVENTS), "moments": len(MOMENTS)}


@app.get("/player/focus")
def player_focus(player: str, minute: int = Query(90, ge=0, le=90), language: str = "en", last: int = 6,
                 speed_kmh: float = Query(30.0, ge=20, le=40), distance_step_km: float = Query(2.0, ge=0.5, le=5),
                 pass_difficulty_min: float = Query(80.0, ge=40, le=85)):
    """Player Focus mode: live summary + threshold alerts (speed, distance, shots, difficult passes) for one player."""
    mine = [e for e in EVENTS if e["player"] == player and e["t"] <= minute * 60]
    if not any(e["player"] == player for e in EVENTS):
        raise HTTPException(404, "player not found")
    ps = [e for e in mine if e["type"] == "pass"]
    ok = [e for e in ps if e["success"]]
    diffs = [pass_difficulty(e) for e in ps if e["end_x"] is not None]
    summary = dict(player=player, passes=len(ps), pass_accuracy_pct=round(100 * len(ok) / max(len(ps), 1)),
                   avg_pass_difficulty=round(sum(diffs) / max(len(diffs), 1), 1), shots=len([e for e in mine if e["type"] == "shot"]),
                   xg=round(sum(e["xg"] or 0 for e in mine), 2), goals=len([e for e in mine if e["outcome"] == "goal"]),
                   tackles=len([e for e in mine if e["type"] == "tackle"]), distance_km=mine[-1]["player_distance_km"] if mine else 0,
                   top_speed_kmh=max([e["player_speed_kmh"] or 0 for e in mine] or [0]))
    alerts = player_alerts(player, minute * 60, 0, speed_kmh, distance_step_km, pass_difficulty_min)[-last:]
    return {"summary": summary, "alerts": translate_cards(alerts, language)}


@app.get("/voiceover/voices")
def voiceover_voices(locale: str = "en-NG"):
    """List the Azure neural voices available in your region for a locale (use it to pick/override voices)."""
    if not (AZURE_KEY and AZURE_REGION):
        raise HTTPException(503, "Azure Speech not configured")
    r = requests.get(f"https://{AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/voices/list",
                     headers={"Ocp-Apim-Subscription-Key": AZURE_KEY}, timeout=20)
    if r.status_code != 200:
        raise HTTPException(502, f"Azure Speech {r.status_code}: {r.text[:200]}")
    return [dict(name=v["ShortName"], gender=v.get("Gender"), locale=v["Locale"]) for v in r.json() if v["Locale"].lower() == locale.lower()]


@app.post("/voiceover")
def voiceover(req: VoiceReq):
    """Azure AI Speech text-to-speech (MP3) for any narration, overlay card or recap."""
    if not (AZURE_KEY and AZURE_REGION):
        raise HTTPException(503, "Azure Speech not configured (set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION)")
    if req.language not in VOICES:
        raise HTTPException(422, f"No Azure neural voice configured for '{req.language}'")
    rate = req.rate if re.fullmatch(r"[+-]?\d{1,2}%", req.rate) else "+0%"
    key = ("tts", req.text, req.language, rate)
    if key in CACHE:
        return Response(CACHE[key], media_type="audio/mpeg")
    voice, loc = os.getenv(f"AZURE_VOICE_{req.language.upper()}", VOICES[req.language][0]), VOICES[req.language][1]
    ssml = (f"<speak version='1.0' xml:lang='{loc}'><voice name='{voice}'><prosody rate='{rate}'>"
            f"{xml_escape(req.text[:1500])}</prosody></voice></speak>")
    try:
        r = requests.post(f"https://{AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
                          headers={"Ocp-Apim-Subscription-Key": AZURE_KEY, "Content-Type": "application/ssml+xml",
                                   "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3", "User-Agent": "PitchLens"},
                          data=ssml.encode("utf-8"), timeout=30)
    except requests.RequestException as ex:
        raise HTTPException(502, f"Azure Speech unreachable: {ex}")
    if r.status_code != 200:
        raise HTTPException(502, f"Azure Speech {r.status_code}: {r.text[:200]}")
    CACHE[key] = r.content
    return Response(r.content, media_type="audio/mpeg")


@app.post("/match/new")
def new_match(seed: int = 7, empty: bool = False):
    """New synthetic match. empty=true starts with no events so a live feed (/autoevent/tracking) can fill it."""
    global EVENTS, MOM, MOMENTS
    EVENTS = [] if empty else generate_match(seed)
    MOM = build_momentum(EVENTS)
    MOMENTS = detect_moments(EVENTS, MOM)
    CACHE.clear()
    return {"seed": seed, "events": len(EVENTS), "moments": len(MOMENTS), "score": compute_state(EVENTS)["score"]}