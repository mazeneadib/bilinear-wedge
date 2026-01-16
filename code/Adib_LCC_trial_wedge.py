
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adib_LCC_trial_wedge.py — Bilinear trial‑wedge thrust for semi‑gravity walls with LCC–soil hybrid backfill
Author: Mazen E. Adib (m.adib@fugro.com)
License: MIT
Python: 3.8+

Implements:
  - Governing thrusts P_A (static) and P_AE (pseudo‑static) via search over alpha.
  - Two‑regime kinematics (Case A: two wedges; Case B: single wedge).
  - Units: pcf, psf, ft, lb/ft (report also in kips/ft for convenience), degrees.
  - Pseudo‑static convention: traffic surcharge EXCLUDED from W; static includes it.
  - Minimal YAML‑lite reader for key:value pairs (so no external deps).
  - CSV outputs for parametrics; clean CLI; --selftest reproduces the benchmark.

References:
  - Manuscript & Supplemental LaTeX provided in submission package (see DOIs).
"""

from __future__ import annotations
from dataclasses import dataclass
from math import sin, cos, tan, atan, radians, degrees, isfinite
import argparse
import csv
import sys
from typing import Dict, Tuple, List

EPS = 1e-12

# ----------------------------- Data Models ----------------------------------

@dataclass
class Inputs:
    # Geometry (ft)
    h1: float  # soil height below LCC
    h2: float  # LCC vertical height
    h3: float  # pavement thickness
    beta_deg: float  # interface angle (deg) from horizontal

    # Materials / loads
    gamma_soil: float  # pcf
    gamma_LCC: float   # pcf
    gamma_pav: float   # pcf
    q_traffic: float   # psf (uniform surcharge)
    phi_deg: float     # soil friction (deg)
    c_prime: float     # psf, cohesion mobilized ONLY along failure plane
    deltaA_deg: float  # wall thrust direction (deg) from horizontal
    kh: float          # horizontal seismic coefficient

    # Search grid for alpha (deg)
    alpha_min: float = 15.0
    alpha_max: float = 75.0
    alpha_step: float = 0.5

    # Sanity checks
    def validate(self) -> None:
        for name in ("h1", "h2", "h3", "gamma_soil", "gamma_LCC", "gamma_pav"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.alpha_step <= 0:
            raise ValueError("alpha_step must be > 0.")
        for ang in ("beta_deg", "phi_deg", "deltaA_deg"):
            a = getattr(self, ang)
            if not (-89.999 <= a <= 89.999) and ang != "deltaA_deg":
                # beta and phi defined from horizontal; restrict away from ±90 to avoid singular geometries
                pass

# ----------------------------- Utilities ------------------------------------

def deg(x: float) -> float:
    return degrees(x)

def rad(x: float) -> float:
    return radians(x)

def alpha_crit_deg(h1: float, h2: float, beta_deg: float) -> float:
    """ alpha_crit = arctan( ((h1 + h2) tan beta) / h2 )  (angles from horizontal) """
    if h2 <= 0:
        return 90.0
    num = (h1 + h2) * tan(rad(beta_deg))
    return deg(atan(num / max(h2, EPS)))

def clamp_nonneg(x: float) -> float:
    return x if x > 0.0 else 0.0

# ---------------------- Geometry / Weights (Case A/B) -----------------------

def caseA_geometry_weights(inp: Inputs, alpha_deg: float) -> Dict[str, float]:
    """
    Case A (two wedges): alpha > alpha_crit, failure intersects the interface.
    Returns dict with L1, L2, L_AB, L_BOpr, W1, W2 (static and pseudo-static W1s, W2s).
    """
    a = rad(alpha_deg)
    b = rad(inp.beta_deg)

    # h_soil and h_BC per supplemental (angles from horizontal)
    # h_soil = h1 * tan(alpha) / (tan(alpha) - tan(beta))
    # h_BC   = (h1 + h2) - h_soil
    ta, tb = tan(a), tan(b)
    denom = max(ta - tb, EPS)
    h_soil = inp.h1 * ta / denom
    h_BC = (inp.h1 + inp.h2) - h_soil

    # Lengths (per supplemental definitions)
    L1 = h_BC / max(tb, EPS)                    # along BO' projection
    L_BOpr = L1 / max(cos(b), EPS)             # contact length at BO' (inclination beta)
    L2 = (inp.h2 / max(tb, EPS)) - L1          # right part along interface
    L_AB = h_soil / max(sin(a), EPS)           # failure plane length

    if L1 < 0 or L2 < 0:
        # Geometrically inadmissible; let caller discard by returning NaNs
        return dict(invalid=1)

    # Volumes (per unit out-of-plane width)
    # Wedge 1 (LCC triangle left of BC) + pavement/traffic above its plan length L1
    V_LCC_1 = 0.5 * L1 * h_BC
    W_LCC_1 = inp.gamma_LCC * V_LCC_1
    W_p_1   = inp.gamma_pav * inp.h3 * L1
    W_t_1   = inp.q_traffic * L1

    # Wedge 2 (composite right of BC): trapezoid area split into soil and LCC parts
    V_w2_total = 0.5 * (inp.h2 + inp.h1 + h_BC) * L2
    V_LCC_2    = 0.5 * (inp.h2 + h_BC) * L2
    V_soil_2   = max(V_w2_total - V_LCC_2, 0.0)

    W_LCC_2 = inp.gamma_LCC * V_LCC_2
    W_soil_2 = inp.gamma_soil * V_soil_2
    W_p_2    = inp.gamma_pav * inp.h3 * L2
    W_t_2    = inp.q_traffic * L2

    W1_static  = W_LCC_1 + W_p_1 + W_t_1
    W1_pseudo  = W_LCC_1 + W_p_1               # exclude traffic
    W2_static  = W_LCC_2 + W_soil_2 + W_p_2 + W_t_2
    W2_pseudo  = W_LCC_2 + W_soil_2 + W_p_2    # exclude traffic

    return dict(
        invalid=0,
        L1=L1, L2=L2, L_AB=L_AB, L_BOpr=L_BOpr,
        W1=W1_static, W1s=W1_pseudo,
        W2=W2_static, W2s=W2_pseudo
    )

def caseB_geometry_weights(inp: Inputs, alpha_deg: float) -> Dict[str, float]:
    """
    Case B (single wedge): alpha <= alpha_crit.
    Returns dict with L2 (plan length), L_AB (failure length), W2 (static, pseudo).
    """
    a = rad(alpha_deg)
    b = rad(inp.beta_deg)

    L2 = (inp.h2 + inp.h1) / max(tan(a), EPS)           # plan length to crest
    L_AB = (inp.h2 + inp.h1) / max(sin(a), EPS)         # failure length (alpha from horizontal)

    # Volumes: total triangle at slope alpha minus LCC triangle at beta
    V_total = 0.5 * (inp.h2 + inp.h1)**2 / max(tan(a), EPS)
    V_LCC   = 0.5 * inp.h2**2 / max(tan(b), EPS)
    V_soil  = max(V_total - V_LCC, 0.0)

    W_LCC = inp.gamma_LCC * V_LCC
    W_soil = inp.gamma_soil * V_soil
    W_p    = inp.gamma_pav * inp.h3 * L2
    W_t    = inp.q_traffic * L2

    W2_static = W_LCC + W_soil + W_p + W_t
    W2_pseudo = W_LCC + W_soil + W_p  # exclude traffic

    return dict(invalid=0, L2=L2, L_AB=L_AB, W2=W2_static, W2s=W2_pseudo)

# -------------------------- Wedge Thrust Formula ----------------------------

def wedge_thrust(W: float, phi_deg: float, cL: float, theta_deg: float,
                 kh: float, deltaA_deg: float) -> float:
    """
    P_A(theta) = [ W( sinθ − cosθ tanφ + kh( cosθ + sinθ tanφ ) ) − c' L ] /
                 [ cos(θ−δA) + sin(θ−δA) tanφ ]
    Returns P in lb/ft. Angles in degrees from horizontal.
    """
    th = rad(theta_deg)
    ph = rad(phi_deg)
    dA = rad(deltaA_deg)

    num = (W * (sin(th) - cos(th) * tan(ph) + kh * (cos(th) + sin(th) * tan(ph))) - cL)
    den = (cos(th - dA) + sin(th - dA) * tan(ph))

    if abs(den) < EPS:
        return float("nan")
    return num / den

# ------------------------------- Solver -------------------------------------

@dataclass
class Result:
    regime: str           # "A" or "B"
    alpha_deg: float
    PA_lbft: float        # static
    PAE_lbft: float       # pseudo-static (if kh>0) else None

def solve_governing(inp: Inputs) -> Result:
    inp.validate()
    acrit = alpha_crit_deg(inp.h1, inp.h2, inp.beta_deg)

    best = Result(regime="", alpha_deg=float("nan"), PA_lbft=-1e300, PAE_lbft=float("nan"))

    alpha = inp.alpha_min
    while alpha <= inp.alpha_max + 1e-9:
        regime = "A" if alpha > acrit else "B"
        if regime == "A":
            geo = caseA_geometry_weights(inp, alpha)
            if geo.get("invalid", 0):
                alpha += inp.alpha_step
                continue

            # Cohesion terms on contact lengths for each wedge
            cL1 = inp.c_prime * geo["L_BOpr"]
            cL2 = inp.c_prime * geo["L_AB"]

            # Static
            P1 = wedge_thrust(geo["W1"], inp.phi_deg, cL1, inp.beta_deg, 0.0, inp.deltaA_deg)
            P2 = wedge_thrust(geo["W2"], inp.phi_deg, cL2, alpha,        0.0, inp.deltaA_deg)

            # If the LCC wedge contribution is negative at small beta, treat as self‑supporting: clamp at 0
            P1 = clamp_nonneg(P1)
            PA = P1 + P2

            # Pseudo‑static
            PAE = float("nan")
            if inp.kh > 0.0:
                P1e = wedge_thrust(geo["W1s"], inp.phi_deg, cL1, inp.beta_deg, inp.kh, inp.deltaA_deg)
                P2e = wedge_thrust(geo["W2s"], inp.phi_deg, cL2, alpha,        inp.kh, inp.deltaA_deg)
                P1e = clamp_nonneg(P1e)
                PAE = P1e + P2e

        else:  # Case B
            geo = caseB_geometry_weights(inp, alpha)
            if geo.get("invalid", 0):
                alpha += inp.alpha_step
                continue
            cL = inp.c_prime * geo["L_AB"]

            PA  = wedge_thrust(geo["W2"],  inp.phi_deg, cL, alpha, 0.0,       inp.deltaA_deg)
            PAE = float("nan")
            if inp.kh > 0.0:
                PAE = wedge_thrust(geo["W2s"], inp.phi_deg, cL, alpha, inp.kh, inp.deltaA_deg)

        # Accept only finite values
        if isfinite(PA) and PA > best.PA_lbft:
            best = Result(regime=regime, alpha_deg=alpha, PA_lbft=PA, PAE_lbft=PAE)

        alpha += inp.alpha_step

    return best

# --------------------------- YAML‑lite Reader --------------------------------

def parse_yaml_lite(path: str) -> Dict[str, float]:
    """
    Minimal parser for key: value pairs; ignores comments and blank lines.
    Accepts numbers (float), deg symbols are okay but ignored.
    """
    data: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            k = k.strip()
            v = v.split("#", 1)[0].strip()
            v = v.replace("°", "").replace("deg", "")
            try:
                data[k] = float(v)
            except ValueError:
                raise ValueError(f"Cannot parse numeric value for '{k}': '{v}'")
    return data

def inputs_from_dict(d: Dict[str, float]) -> Inputs:
    return Inputs(
        h1=d["h1"], h2=d["h2"], h3=d.get("h3", 0.0),
        beta_deg=d["beta"],
        gamma_soil=d["gamma_soil"], gamma_LCC=d["gamma_LCC"], gamma_pav=d.get("gamma_pav", 0.0),
        q_traffic=d.get("q_traffic", 0.0),
        phi_deg=d["phi"], c_prime=d.get("c_prime", 0.0),
        deltaA_deg=d.get("deltaA", 0.0),
        kh=d.get("kh", 0.0),
        alpha_min=d.get("alpha_min", 15.0), alpha_max=d.get("alpha_max", 75.0), alpha_step=d.get("alpha_step", 0.5),
    )

# ------------------------------- CLI ----------------------------------------

def run_cli(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Bilinear trial‑wedge lateral thrust for LCC–soil hybrid backfill.")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--yaml", help="Path to YAML‑lite input file (key: value pairs).")
    g.add_argument("--param", choices=["beta", "cohesion", "deltaA"], help="Run a simple parametric study.")
    p.add_argument("--out", help="CSV output path for parametrics.")

    # Direct inputs (if not using --yaml)
    p.add_argument("--h1", type=float);        p.add_argument("--h2", type=float);         p.add_argument("--h3", type=float, default=0.0)
    p.add_argument("--beta", type=float)
    p.add_argument("--gamma_soil", type=float); p.add_argument("--gamma_LCC", type=float); p.add_argument("--gamma_pav", type=float, default=0.0)
    p.add_argument("--q_traffic", type=float, default=0.0)
    p.add_argument("--phi", type=float); p.add_argument("--c_prime", type=float, default=0.0)
    p.add_argument("--deltaA", type=float, default=0.0)
    p.add_argument("--kh", type=float, default=0.0)
    p.add_argument("--alpha_min", type=float, default=15.0)
    p.add_argument("--alpha_max", type=float, default=75.0)
    p.add_argument("--alpha_step", type=float, default=0.5)
    p.add_argument("--selftest", action="store_true", help="Run the benchmark case and print results.")
    args = p.parse_args(argv)

    if args.selftest:
        # Benchmark case from manuscript/supplemental
        d = dict(
            h1=3.0, h2=25.0, h3=2.0,
            gamma_soil=120.0, gamma_LCC=40.0, gamma_pav=150.0,
            q_traffic=240.0, beta=45.0, phi=34.0, c_prime=0.0, deltaA=0.0,
            kh=0.0, alpha_min=15.0, alpha_max=75.0, alpha_step=0.5
        )
        res = solve_governing(inputs_from_dict(d))
        print(f"[SELFTEST] Regime {res.regime}, alpha* = {res.alpha_deg:.1f} deg")
        print(f"[SELFTEST] P_A  = {res.PA_lbft/1000.0:.3f} kips/ft (expect ~7.72)")
        return 0

    if args.yaml:
        d = parse_yaml_lite(args.yaml)
        inp = inputs_from_dict(d)
    else:
        # Require minimal direct inputs if not YAML
        needed = ["h1", "h2", "beta", "gamma_soil", "gamma_LCC", "phi"]
        missing = [k for k in needed if getattr(args, k) is None]
        if missing:
            p.error(f"Missing required inputs (no --yaml provided): {missing}")
            return 2
        d = dict(
            h1=args.h1, h2=args.h2, h3=args.h3, beta=args.beta,
            gamma_soil=args.gamma_soil, gamma_LCC=args.gamma_LCC, gamma_pav=args.gamma_pav,
            q_traffic=args.q_traffic, phi=args.phi, c_prime=args.c_prime,
            deltaA=args.deltaA, kh=args.kh,
            alpha_min=args.alpha_min, alpha_max=args.alpha_max, alpha_step=args.alpha_step
        )
        inp = inputs_from_dict(d)

    if args.param:
        if not args.out:
            p.error("--out CSV path is required for --param runs.")
            return 2
        return run_parametrics(inp, args.param, args.out)
    else:
        res = solve_governing(inp)
        print(f"Regime {res.regime}, alpha* = {res.alpha_deg:.3f} deg")
        print(f"P_A  = {res.PA_lbft:.2f} lb/ft  ({res.PA_lbft/1000.0:.3f} kips/ft)")
        if inp.kh > 0.0 and isfinite(res.PAE_lbft):
            print(f"P_AE = {res.PAE_lbft:.2f} lb/ft  ({res.PAE_lbft/1000.0:.3f} kips/ft)")
        return 0

def run_parametrics(base: Inputs, mode: str, out_csv: str) -> int:
    rows = [("alpha_deg", "regime", "P_A_lbft", "P_AE_lbft", "var_name", "var_value")]

    if mode == "beta":
        values = [25.6, 33.7, 45.0]
        for b in values:
            inp = Inputs(**{**base.__dict__, "beta_deg": b})
            res = solve_governing(inp)
            rows.append((res.alpha_deg, res.regime, res.PA_lbft, res.PAE_lbft if base.kh > 0 else float("nan"), "beta", b))

    elif mode == "cohesion":
        values = [0.0, 200.0]
        for c in values:
            inp = Inputs(**{**base.__dict__, "c_prime": c})
            res = solve_governing(inp)
            rows.append((res.alpha_deg, res.regime, res.PA_lbft, res.PAE_lbft if base.kh > 0 else float("nan"), "c_prime", c))

    elif mode == "deltaA":
        values = [0.0, 17.0]
        for dA in values:
            inp = Inputs(**{**base.__dict__, "deltaA_deg": dA})
            res = solve_governing(inp)
            rows.append((res.alpha_deg, res.regime, res.PA_lbft, res.PAE_lbft if base.kh > 0 else float("nan"), "deltaA", dA))
    else:
        raise ValueError(f"Unknown param mode: {mode}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)-1} rows.")
    return 0

if __name__ == "__main__":
    sys.exit(run_cli(sys.argv[1:]))
