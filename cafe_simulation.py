import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
import matplotlib.patheffects as pe
import time
import random
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import defaultdict

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulasi Kafe",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1a1a1a; }
.metric-card { background: #f8f6f0; border-radius:10px; padding:14px 18px; margin:4px 0; border-left:3px solid #8B5E3C; }
.metric-val  { font-size:28px; font-weight:700; color:#3d2b1f; }
.metric-lbl  { font-size:12px; color:#7a6050; margin-top:2px; }
.log-entry   { font-size:12px; font-family:monospace; padding:2px 0; border-bottom:1px solid #eee; }
.log-arrive  { color:#1D9E75; }
.log-reject  { color:#E24B4A; }
.log-leave   { color:#BA7517; }
.log-depart  { color:#888; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
OPEN_H   = 8.0
CLOSE_H  = 22.5
N_TABLES = 40

# Tables 1-18 and 37-40 have 2 seats; 19-36 have 1 seat per side (paired)
TABLE_SEATS = {i: 2 for i in range(1, 41)}
for i in range(19, 37):
    TABLE_SEATS[i] = 1  # single-side, paired across divider

# ─── Layout coordinates (matching hand-drawn floorplan) ───────────────────────
def build_layout():
    """Returns dict: table_id -> (cx, cy, seat_coords_list)"""
    layout = {}
    TW, TH = 0.55, 0.55   # table dims in data coords
    SR = 0.22              # seat radius offset

    def add(tid, x, y, seat_positions):
        layout[tid] = {"x": x, "y": y, "w": TW, "h": TH, "seats": seat_positions}

    # ── Column 1: single-column tables 37-40 (left wall) ──
    col1_x = 0.5
    for idx, tid in enumerate([37, 38, 39, 40]):
        y = 9.0 - idx * 1.8
        add(tid, col1_x, y, [(col1_x - SR, y + TH/2), (col1_x + TW + SR, y + TH/2)])

    # ── Middle section: paired tables 19-36 (with divider) ──
    pairs = [(36,19),(35,20),(34,21),(33,22),(32,23),(31,24),(30,25),(29,26),(28,27)]
    mx = 2.2
    for idx, (left_id, right_id) in enumerate(pairs):
        y = 9.0 - idx * 0.95
        add(left_id,  mx,        y, [(mx - SR, y + TH/2)])
        add(right_id, mx + TW + 0.15, y, [(mx + TW + 0.15 + TW + SR, y + TH/2)])

    # ── Column 3: tables 12, 14, 16, 18 (left) and 13, 15, 17 (right) ──
    c3x = 4.8
    for idx, (a, b) in enumerate([(12, None), (14, 13), (16, 15), (18, 17)]):
        y = 9.0 - idx * 1.7
        add(a, c3x, y, [(c3x - SR, y + TH/2)])
        if b:
            add(b, c3x + TW + 0.4, y, [(c3x + TW + 0.4 + TW + SR, y + TH/2)])

    # ── Column 4: tables 5-11 (two seats each) ──
    c4x = 7.0
    for idx, tid in enumerate([5, 6, 7, 8, 9, 10, 11]):
        y = 9.0 - idx * 1.2
        add(tid, c4x, y, [(c4x - SR, y + TH/2), (c4x + TW + SR, y + TH/2)])

    # ── Column 5: tables 1-4 (two seats each) ──
    c5x = 9.0
    for idx, tid in enumerate([4, 3, 2, 1]):
        y = 9.0 - idx * 1.7
        add(tid, c5x, y, [(c5x - SR, y + TH/2), (c5x + TW + SR, y + TH/2)])

    return layout

LAYOUT = build_layout()

# ─── Simulation State ─────────────────────────────────────────────────────────
@dataclass
class TableState:
    occupied: bool = False
    group_id: Optional[str] = None
    leave_time: float = 0.0    # simulation minutes
    group_size: int = 0

@dataclass
class VisitorParticle:
    vid: str
    table_id: Optional[int]
    x: float
    y: float
    tx: float
    ty: float
    color: str
    seated: bool = False
    leaving: bool = False
    exit_x: float = 0.0
    exit_y: float = 5.0
    alpha: float = 1.0

@dataclass
class SimState:
    time_min: float = OPEN_H * 60    # minutes since midnight
    tables: Dict[int, TableState] = field(default_factory=lambda: {i: TableState() for i in range(1, 41)})
    visitors: List[VisitorParticle] = field(default_factory=list)
    total_arrivals: int = 0
    total_rejected: int = 0
    total_left_crowded: int = 0
    total_split: int = 0
    total_served: int = 0
    accum_sec: float = 0.0
    log: List[dict] = field(default_factory=list)
    # Snapshots for time-series
    occ_history: List[tuple] = field(default_factory=list)   # (time_min, occ_count)
    arrive_history: List[tuple] = field(default_factory=list)
    reject_history: List[tuple] = field(default_factory=list)

COLORS = ["#1D9E75","#D85A30","#7F77DD","#D4537E","#BA7517",
          "#185FA5","#639922","#993556","#0F6E56","#993C1D"]

def get_occupancy(tables):
    return sum(1 for t in tables.values() if t.occupied)

def get_arrival_rate(sim_time_min, base_rate):
    """Poisson rate modulated by time of day"""
    h = sim_time_min / 60
    if   h < 9:   mult = 0.3
    elif h < 10:  mult = 0.5
    elif h < 11:  mult = 0.8
    elif h < 12:  mult = 1.0
    elif h < 14:  mult = 1.3
    elif h < 16:  mult = 1.0
    elif h < 18:  mult = 0.9
    elif h < 20:  mult = 0.8
    elif h < 21:  mult = 0.6
    else:         mult = 0.35
    return base_rate * mult

def find_free_tables(tables, count):
    free = [tid for tid, t in tables.items() if not t.occupied]
    if not free:
        return None
    if count == 1:
        return [random.choice(free)]
    # Try to find adjacent tables
    if count >= 2:
        random.shuffle(free)
        for tid in free:
            t_pos = LAYOUT[tid]
            adj = []
            for other_id in free:
                if other_id == tid: continue
                o_pos = LAYOUT[other_id]
                dist = math.sqrt((t_pos["x"]-o_pos["x"])**2 + (t_pos["y"]-o_pos["y"])**2)
                if dist < 2.5:
                    adj.append(other_id)
            if adj:
                selected = [tid] + adj[:count-1]
                if len(selected) >= count:
                    return selected[:count]
        if len(free) >= count:
            return free[:count]
    return None if len(free) < count else free[:count]

def spawn_visitor(state: SimState, stay_mean_min: float, leave_prob: float):
    occ = get_occupancy(state.tables)
    pct = occ / N_TABLES
    state.total_arrivals += 1

    def add_log(msg, typ):
        hh = int(state.time_min // 60) % 24
        mm = int(state.time_min % 60)
        state.log.insert(0, {"time": f"{hh:02d}:{mm:02d}", "msg": msg, "type": typ})
        if len(state.log) > 60:
            state.log.pop()

    # Full → reject
    if pct >= 1.0:
        state.total_rejected += 1
        add_log("Kafe penuh — pengunjung ditolak!", "reject")
        _add_rejected_walker(state)
        return

    # 90%+ crowded → probabilistic leave
    if pct >= 0.9 and random.random() < leave_prob:
        state.total_left_crowded += 1
        add_log(f"Pengunjung pergi — terlalu ramai ({pct:.0%} penuh)", "leave")
        _add_rejected_walker(state)
        return

    # Group size: mostly solo/pairs
    r = random.random()
    if   r < 0.50: group_size = 1
    elif r < 0.75: group_size = 2
    elif r < 0.88: group_size = 3
    else:          group_size = 4

    stay_dur = max(30, np.random.normal(stay_mean_min, stay_mean_min * 0.25))
    gid = f"g{state.total_arrivals}"

    # Groups >2 may split (40% chance)
    will_split = group_size > 2 and random.random() < 0.40

    if will_split:
        tables_needed = group_size
        selected = find_free_tables(state.tables, tables_needed)
        if not selected:
            state.total_rejected += 1
            add_log(f"Grup {group_size} orang tidak ada meja", "reject")
            _add_rejected_walker(state)
            return
        state.total_split += 1
        add_log(f"Grup {group_size} pisah ke meja {', '.join(map(str,selected))}", "arrive")
        for tid in selected:
            state.tables[tid].occupied = True
            state.tables[tid].group_id = gid
            state.tables[tid].leave_time = state.time_min + stay_dur
            state.tables[tid].group_size = 1
            _add_seated_visitor(state, tid)
    else:
        selected = find_free_tables(state.tables, 1)
        if not selected:
            state.total_rejected += 1
            add_log("Tidak ada meja kosong", "reject")
            _add_rejected_walker(state)
            return
        tid = selected[0]
        state.tables[tid].occupied = True
        state.tables[tid].group_id = gid
        state.tables[tid].leave_time = state.time_min + stay_dur
        state.tables[tid].group_size = group_size
        lbl = "1 orang" if group_size == 1 else f"Grup {group_size}"
        add_log(f"{lbl} duduk di meja {tid}", "arrive")
        _add_seated_visitor(state, tid)
        state.total_served += 1

def _add_seated_visitor(state: SimState, tid: int):
    t = LAYOUT[tid]
    cx = t["x"] + t["w"] / 2
    cy = t["y"] + t["h"] / 2
    entry_x = random.choice([-0.5, 10.5])
    color = random.choice(COLORS)
    lv = state.tables[tid].leave_time
    state.visitors.append(VisitorParticle(
        vid=f"v{len(state.visitors)}{random.random():.4f}",
        table_id=tid,
        x=entry_x, y=random.uniform(1, 9),
        tx=cx, ty=cy,
        color=color,
        exit_x=entry_x, exit_y=cy,
    ))

def _add_rejected_walker(state: SimState):
    entry_x = random.choice([-0.5, 10.5])
    exit_x  = 10.5 if entry_x < 5 else -0.5
    y = random.uniform(2, 8)
    state.visitors.append(VisitorParticle(
        vid=f"rej{random.random():.4f}",
        table_id=None,
        x=entry_x, y=y,
        tx=exit_x, ty=y,
        color="#E24B4A",
        alpha=0.6,
        leaving=True,
        exit_x=exit_x, exit_y=y,
    ))

def check_departures(state: SimState):
    hh = int(state.time_min // 60) % 24
    mm = int(state.time_min % 60)
    for tid, tbl in state.tables.items():
        if tbl.occupied and state.time_min >= tbl.leave_time:
            tbl.occupied = False
            tbl.group_id = None
            state.log.insert(0, {"time": f"{hh:02d}:{mm:02d}",
                                  "msg": f"Meja {tid} kini kosong", "type": "depart"})
            for v in state.visitors:
                if v.table_id == tid and not v.leaving:
                    v.leaving = True
                    v.tx = v.exit_x
                    v.ty = v.exit_y

def update_visitors(state: SimState, dt_sim_min: float):
    speed = 8.0  # units per simulated minute
    new_list = []
    for v in state.visitors:
        dx, dy = v.tx - v.x, v.ty - v.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0.05:
            step = min(dist, speed * dt_sim_min)
            v.x += dx / dist * step
            v.y += dy / dist * step
        else:
            v.x, v.y = v.tx, v.ty
            if not v.leaving:
                v.seated = True
        if v.leaving:
            if abs(v.x - v.exit_x) < 0.2 and abs(v.y - v.exit_y) < 0.2:
                continue  # remove
        new_list.append(v)
    state.visitors = new_list

# ─── Drawing ──────────────────────────────────────────────────────────────────
def draw_floorplan(state: SimState):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#1a1814")
    ax.set_facecolor("#1a1814")
    ax.set_xlim(-0.8, 11.2)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    occ = get_occupancy(state.tables)
    pct = occ / N_TABLES

    # ── Divider line between tables 19-36 ──
    ax.plot([2.75, 2.75], [-0.3, 9.5], color="#444", linewidth=1.5, alpha=0.6, zorder=1)

    # ── Barista station (diamond) ──
    bx, by = 10.2, 9.5
    diamond = Polygon([(bx, by+0.35), (bx+0.7, by), (bx, by-0.35), (bx-0.7, by)],
                      closed=True, facecolor="#1a3a4a", edgecolor="#4a9ecc",
                      linewidth=1.2, zorder=4)
    ax.add_patch(diamond)
    ax.text(bx, by, "☕", ha="center", va="center", fontsize=8, zorder=5)

    # ── Draw tables ──
    for tid in range(1, 41):
        if tid not in LAYOUT: continue
        t = LAYOUT[tid]
        tbl_state = state.tables[tid]
        occ_t = tbl_state.occupied

        face = "#1e3a1e" if occ_t else "#2a2a26"
        edge = "#1D9E75" if occ_t else "#555550"
        lw   = 1.5 if occ_t else 0.7

        rect = FancyBboxPatch((t["x"], t["y"]), t["w"], t["h"],
                               boxstyle="round,pad=0.04",
                               facecolor=face, edgecolor=edge, linewidth=lw, zorder=3)
        ax.add_patch(rect)
        ax.text(t["x"] + t["w"]/2, t["y"] + t["h"]/2, str(tid),
                ha="center", va="center", fontsize=5.5, color="#1D9E75" if occ_t else "#666",
                fontweight="bold" if occ_t else "normal", zorder=4)

        # Seats
        for sx, sy in t["seats"]:
            sc = "#2a6a2a" if occ_t else "#333330"
            se = "#1D9E75" if occ_t else "#444"
            seat = Circle((sx, sy), 0.18, facecolor=sc, edgecolor=se, linewidth=0.7, zorder=3)
            ax.add_patch(seat)

    # ── Visitors ──
    for v in state.visitors:
        size = 28 if v.seated else 20
        ax.scatter(v.x, v.y, s=size, color=v.color, alpha=v.alpha, zorder=5,
                   edgecolors="white", linewidths=0.5)
        if v.seated and not v.leaving:
            ax.scatter(v.x, v.y, s=size*3, color=v.color, alpha=0.15, zorder=4)

    # ── Occupancy overlay text ──
    occ_color = "#639922" if pct < 0.7 else "#BA7517" if pct < 0.9 else "#E24B4A"
    hh = int(state.time_min // 60) % 24
    mm = int(state.time_min % 60)
    ax.text(0.02, 0.98, f"{hh:02d}:{mm:02d}", transform=ax.transAxes,
            fontsize=13, color="#ddd", va="top", fontweight="bold")
    ax.text(0.98, 0.98, f"{pct:.0%} penuh  ({occ}/{N_TABLES} meja)",
            transform=ax.transAxes, fontsize=10, color=occ_color,
            va="top", ha="right")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#1e3a1e", edgecolor="#1D9E75", label="Terisi"),
        mpatches.Patch(facecolor="#2a2a26", edgecolor="#555550", label="Kosong"),
        mpatches.Patch(facecolor="#E24B4A", label="Ditolak/Pergi"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=7,
              facecolor="#222", edgecolor="#444", labelcolor="white",
              framealpha=0.8)

    plt.tight_layout(pad=0.3)
    return fig

def draw_charts(state: SimState):
    if not state.occ_history:
        return None
    df = pd.DataFrame(state.occ_history, columns=["time_min", "occ"])
    df["hour"] = df["time_min"] / 60
    df["pct"] = df["occ"] / N_TABLES * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    fig.patch.set_facecolor("#1a1814")

    for axi in axes:
        axi.set_facecolor("#1a1814")
        axi.spines[["top","right"]].set_visible(False)
        axi.spines[["left","bottom"]].set_color("#444")
        axi.tick_params(colors="#888", labelsize=8)
        axi.xaxis.label.set_color("#888")
        axi.yaxis.label.set_color("#888")

    # Occupancy over time
    axes[0].fill_between(df["hour"], df["pct"], alpha=0.3, color="#1D9E75")
    axes[0].plot(df["hour"], df["pct"], color="#1D9E75", linewidth=1.5)
    axes[0].axhline(90, color="#BA7517", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[0].axhline(100, color="#E24B4A", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[0].set_xlim(OPEN_H, CLOSE_H)
    axes[0].set_ylim(0, 105)
    axes[0].set_xlabel("Jam", fontsize=8)
    axes[0].set_ylabel("% Penuh", fontsize=8)
    axes[0].set_title("Tingkat Okupansi", color="#ccc", fontsize=9)
    axes[0].text(OPEN_H+0.1, 91, "90%", color="#BA7517", fontsize=7)

    # Arrivals vs rejections binned by hour
    if state.arrive_history or state.reject_history:
        bins = np.arange(OPEN_H, CLOSE_H + 1)
        arr_hours = [t/60 for t, _ in state.arrive_history]
        rej_hours = [t/60 for t, _ in state.reject_history]
        if arr_hours:
            axes[1].hist(arr_hours, bins=bins, color="#1D9E75", alpha=0.7, label="Datang")
        if rej_hours:
            axes[1].hist(rej_hours, bins=bins, color="#E24B4A", alpha=0.7, label="Ditolak")
        axes[1].set_xlim(OPEN_H, CLOSE_H)
        axes[1].set_xlabel("Jam", fontsize=8)
        axes[1].set_ylabel("Jumlah", fontsize=8)
        axes[1].set_title("Kedatangan vs Ditolak per Jam", color="#ccc", fontsize=9)
        axes[1].legend(fontsize=7, facecolor="#222", edgecolor="#444", labelcolor="white")

    plt.tight_layout(pad=0.5)
    return fig

# ─── Session State Init ───────────────────────────────────────────────────────
if "sim" not in st.session_state:
    st.session_state.sim = SimState()
if "running" not in st.session_state:
    st.session_state.running = False
if "speed" not in st.session_state:
    st.session_state.speed = 5

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☕ Parameter Simulasi")
    st.markdown("---")

    arrival_rate = st.slider("Laju Kedatangan (tamu/jam)", 1, 25, 8, 1,
                              help="Rata-rata tamu per jam pada jam puncak")
    stay_hours = st.slider("Rata-rata Durasi Duduk (jam)", 0.5, 5.0, 3.0, 0.5,
                            help="Distribusi normal dengan std 25%")
    leave_pct = st.slider("Prob. Pergi karena Ramai (%)", 0, 60, 20, 5,
                           help="Peluang pergi saat kafe ≥90% penuh")

    st.markdown("---")
    st.markdown("### Kecepatan Simulasi")
    speed = st.select_slider("Percepatan", [1, 2, 5, 10, 20, 60], value=st.session_state.speed)
    st.session_state.speed = speed

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Mulai" if not st.session_state.running else "⏸ Pause",
                               use_container_width=True)
    with col2:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    if start_btn:
        st.session_state.running = not st.session_state.running

    if reset_btn:
        st.session_state.sim = SimState()
        st.session_state.running = False
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Asumsi Model:**
    - Kedatangan: **Poisson** (termodulasi waktu)
    - Durasi: **Normal** (mean ± 25%)
    - Grup >2 orang: 40% pisah meja
    - Meja ≥90% penuh → probabilistic leave
    - Meja 1–18, 37–40: 2 kursi
    - Meja 19–36: 1 kursi (berpasangan)
    """)

# ─── Main Layout ──────────────────────────────────────────────────────────────
st.title("☕ Simulasi Ketersediaan Tempat Duduk Kafe")

state: SimState = st.session_state.sim
hh = int(state.time_min // 60) % 24
mm = int(state.time_min % 60)
occ = get_occupancy(state.tables)
pct = occ / N_TABLES

# Metrics row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("🕐 Waktu", f"{hh:02d}:{mm:02d}")
m2.metric("📍 Meja Terisi", f"{occ}/{N_TABLES}", f"{pct:.0%} penuh")
m3.metric("✅ Total Datang", state.total_arrivals)
m4.metric("❌ Ditolak", state.total_rejected,
          f"{state.total_rejected/max(1,state.total_arrivals):.0%}")
m5.metric("😤 Pergi (ramai)", state.total_left_crowded)
m6.metric("🔀 Pisah Meja", state.total_split)

# Floor plan + log
col_map, col_log = st.columns([3, 1])

with col_map:
    fig_floor = draw_floorplan(state)
    floor_placeholder = st.pyplot(fig_floor, use_container_width=True)
    plt.close(fig_floor)

with col_log:
    st.markdown("**📋 Log Kejadian**")
    log_html = ""
    for entry in state.log[:25]:
        cls = {"arrive":"log-arrive","reject":"log-reject",
               "leave":"log-leave","depart":"log-depart"}.get(entry["type"],"")
        icon = {"arrive":"→","reject":"✗","leave":"↩","depart":"←"}.get(entry["type"],"·")
        log_html += f'<div class="log-entry {cls}">[{entry["time"]}] {icon} {entry["msg"]}</div>'
    st.markdown(f'<div style="height:360px;overflow-y:auto;background:#111;padding:8px;border-radius:6px">{log_html}</div>',
                unsafe_allow_html=True)

# Charts
st.markdown("---")
fig_charts = draw_charts(state)
if fig_charts:
    st.pyplot(fig_charts, use_container_width=True)
    plt.close(fig_charts)
else:
    st.info("Grafik akan muncul setelah simulasi berjalan beberapa saat...")

# ─── Simulation Step ──────────────────────────────────────────────────────────
if st.session_state.running:
    sim = state
    TICK_REAL_SEC = 0.3      # real seconds between ticks
    dt_sim = TICK_REAL_SEC * st.session_state.speed * 1.0  # sim minutes per tick

    sim.time_min += dt_sim
    if sim.time_min >= CLOSE_H * 60:
        sim.time_min = OPEN_H * 60
        st.session_state.running = False

    check_departures(sim)

    rate = get_arrival_rate(sim.time_min, arrival_rate)
    sim.accum_sec += dt_sim
    expected = rate * sim.accum_sec / 60
    if expected >= 1:
        n = np.random.poisson(expected)
        for _ in range(min(n, 4)):
            spawn_visitor(sim, stay_hours * 60, leave_pct / 100)
            sim.arrive_history.append((sim.time_min, 1))
            if sim.total_rejected > len(sim.reject_history):
                sim.reject_history.append((sim.time_min, 1))
        sim.accum_sec = 0

    update_visitors(sim, dt_sim)

    # Record snapshot every ~30 sim minutes
    if not sim.occ_history or (sim.time_min - sim.occ_history[-1][0]) >= 15:
        sim.occ_history.append((sim.time_min, get_occupancy(sim.tables)))

    time.sleep(TICK_REAL_SEC)
    st.rerun()
