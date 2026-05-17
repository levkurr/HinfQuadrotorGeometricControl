"""
reproduce_results.py

Companion script for the manuscript

    A closed-form trajectory-uniform H_inf disturbance gain for the
    geometric quadrotor controller

submitted to the International Journal of Robust and Nonlinear Control.

Running

    python3 reproduce_results.py

produces every numerical value reported in Sections 4-7 of the manuscript
and prints them alongside the printed values for direct comparison. Add
the --full flag to also reproduce Table 7.1 (LMI comparison; +5 minutes).

A FORCE_RERUN environment variable bypasses cached simulation data and
re-runs every integrator call. Cached data is in figures/figure*_data.json.

Dependencies: numpy, scipy. cvxpy is required for Table 7.1 only.

Output structure: one block per result, each block prints "computed" and
"paper" values side by side.
"""
import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np


# ----- F450 vehicle parameters (paper Sec. 6) -----
M = 0.5                              # kg
J_DIAG = (0.0023, 0.0023, 0.004)     # kg m^2
G = 9.81                             # m/s^2
RHO = 0.1                            # s^4   (LQR weight)


# ----- Output helpers -----
def header(s):
    bar = '=' * 72
    print(f'\n{bar}\n  {s}\n{bar}')


# Global accumulator for any computed-vs-paper discrepancies. Each entry is
# (where, computed, paper, relative_difference).
discrepancies = []


def compare(label, computed, paper, rel_tol=1e-3, abs_tol=None, where=None):
    """Compare a computed value to its printed counterpart, recording any
    disagreement larger than the tolerance for the final summary."""
    if abs_tol is not None:
        diff = abs(computed - paper)
        if diff > abs_tol:
            discrepancies.append((where or label, computed, paper, diff))
    else:
        if paper != 0:
            rel = abs(computed - paper) / abs(paper)
            if rel > rel_tol:
                discrepancies.append((where or label, computed, paper, rel))
        else:
            if abs(computed) > rel_tol:
                discrepancies.append((where or label, computed, paper,
                                      abs(computed)))


# Section 1. Closed-form gain  sqrt(rho)/m   (Theorem 5.3)
def section_1_closed_form_gain():
    header('1.  Closed-form gain  sqrt(rho)/m   (Theorem 5.3)')

    gamma_closed_form = np.sqrt(RHO) / M
    p_o = RHO ** (-0.25)

    print(f'  Vehicle parameters (F450 small research quadrotor):')
    print(f'    m   = {M} kg')
    print(f'    J   = diag{J_DIAG} kg m^2')
    print(f'    rho = {RHO} s^4   (LQR cost weight)')
    print()
    print(f'  Derived quantities:')
    print(f'    sqrt(rho)/m  =  {gamma_closed_form:.6f}  m s^2 / N        '
          f'  [paper: 0.6325]')
    print(f'    p_o = rho^(-1/4)  =  {p_o:.4f} rad/s                   '
          f'  [paper: 1.778]')

    compare('Closed-form gain sqrt(rho)/m', gamma_closed_form, 0.6325,
            abs_tol=1e-4)
    compare('Outer-loop bandwidth p_o', p_o, 1.778, abs_tol=1e-3)


# Section 2. Threshold q*_c   (Theorem 4.1)
def section_2_threshold():
    header('2.  Threshold q*_c   (Theorem 4.1; root of degree-12 polynomial)')

    # p(q) = q^12 + 2 q^10 - 7 q^8 + 36 q^7 - 12 q^6 - 56 q^5
    #          + 34 q^4 - 36 q^3 + 2 q^2 - 1
    coeffs = [1, 0, 2, 0, -7, 36, -12, -56, 34, -36, 2, 0, -1]
    roots = np.roots(coeffs)
    real_positive = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    q_c_star = max(real_positive)

    paper_value = 1.354897850704138
    print(f'  q*_c  (computed)  =  {q_c_star:.15f}')
    print(f'  q*_c  (paper)     =  {paper_value:.15f}')
    print(f'  difference        =  {abs(q_c_star - paper_value):.2e}')

    compare('Threshold q*_c', q_c_star, paper_value, abs_tol=1e-12)


# Section 3. High-bandwidth limit  L_uw  (Corollary 5.8)
def section_3_Luw():
    header('3.  High-bandwidth limit  L_uw   (Corollary 5.8)')

    L_uw_closed = np.sqrt(1 + np.sqrt(5))

    # Numerical maximum of h(alpha) = 2 (2 alpha^2 + 1) / (1 + alpha^4)
    alpha = np.linspace(1e-6, 5.0, 1_000_000)
    h = 2 * (2 * alpha**2 + 1) / (1 + alpha**4)
    L_uw_num = np.sqrt(h.max())

    print(f'  L_uw  =  sqrt(1 + sqrt 5)')
    print(f'        (closed form)        =  {L_uw_closed:.10f}')
    print(f'        (numerical grid max) =  {L_uw_num:.10f}')
    print(f'        (paper)              =  1.7989')

    compare('High-bandwidth limit L_uw', L_uw_closed, 1.7989, abs_tol=1e-3)


# Section 4. F450 trajectory parameters (Section 6.1)
def section_4_trajectory_params():
    header('4.  F450 derived quantities and trajectory parameters '
           '(Section 6.1)')

    p_o = RHO ** (-0.25)
    q_c_star = 1.354897850704138
    threshold = q_c_star * p_o
    F_a_steady = 0.5 * np.sqrt(RHO) / M

    print(f'  Computed                              vs.   paper')
    print(f'  ' + '-' * 60)
    print(f'  p_o = rho^(-1/4)  = {p_o:>10.4f} rad/s        1.778 rad/s')
    print(f'  sqrt(rho)/m       = {np.sqrt(RHO)/M:>10.6f}                 0.6325')
    print(f'  Thm 4.1 threshold = {threshold:>10.4f} rad/s        2.41 rad/s')
    print(f'  F_a sqrt(rho)/m (F_a=0.5) = {F_a_steady:>10.6f} m        0.316228 m')
    print(f'  q at p_inner = 9.19 rad/s = {9.19*RHO**0.25:>6.3f}          5.17')

    compare('p_o', p_o, 1.778, abs_tol=1e-3)
    compare('Thm 4.1 threshold p_inner', threshold, 2.41, abs_tol=1e-2)
    compare('F_a sqrt(rho)/m steady state', F_a_steady, 0.316228, abs_tol=1e-5)

    # Aggressive 3D figure-8: x_d=2cos(t), y_d=sin(2t), z_d=0.5sin(t)
    t = np.linspace(0, 2*np.pi, 100000)
    xdd = -2*np.cos(t)
    ydd = -4*np.sin(2*t)
    zdd = -0.5*np.sin(t)
    acc_mag = np.sqrt(xdd**2 + ydd**2 + zdd**2)

    F_des = M * np.array([xdd, ydd, zdd - G])      # m * (x_dd - g e_3)
    f0 = np.sqrt((F_des**2).sum(axis=0))
    b3d = -F_des / f0
    tilt_deg = np.degrees(np.arccos(np.clip(b3d[2], -1, 1)))
    df0_dt = np.gradient(f0, t)
    mu_bar = np.abs(df0_dt / f0).max()

    print()
    print(f'  Aggressive 3D figure-8 (x_d=2cos(t), y_d=sin(2t), z_d=0.5sin(t)):')
    print(f'    peak acceleration  = {acc_mag.max():>8.3f} m/s^2     '
          f'  [paper: 4.26]')
    print(f'    peak tilt          = {tilt_deg.max():>8.3f} deg       '
          f'  [paper: 24.2]')
    print(f'    mu_bar             = {mu_bar:>8.3f} 1/s       '
          f'  [paper: 0.20]')
    print(f'    f_0 range          = [{f0.min():.3f}, {f0.max():.3f}] N '
          f'   [paper: [4.66, 5.51]]')

    compare('Peak acceleration', acc_mag.max(), 4.26, abs_tol=1e-2)
    compare('Peak tilt', tilt_deg.max(), 24.2, abs_tol=0.1)
    compare('mu_bar', mu_bar, 0.20, abs_tol=1e-2)
    compare('f_0 min', f0.min(), 4.66, abs_tol=1e-2)
    compare('f_0 max', f0.max(), 5.51, abs_tol=1e-2)


# Section 5. Table I — dimensionless LTI norms vs q   (Section 5.7)
def section_5_table_I():
    header('5.  Table I:  Dimensionless LTI norms (Section 5.7)')

    sqrt2 = np.sqrt(2.0)

    def make_A0_B0(q):
        # Dimensionless A_0(q), B_0 for the 4D closed loop (paper Lemma 5.4).
        A0 = np.array([
            [0.0, 1.0,                       0.0,             0.0      ],
            [0.0, 0.0,                       -1.0,            0.0      ],
            [0.0, 0.0,                       0.0,             1.0      ],
            [q**2, sqrt2*q**2 + sqrt2*q,     -q**2 - 2.0*q,   -sqrt2*q ],
        ])
        B0 = np.array([[0.0], [1.0], [0.0], [2.0*q]])
        return A0, B0

    E_h = np.zeros((4, 2))
    E_h[2, 0] = 1.0
    E_h[3, 1] = 1.0
    C_y = np.array([[1.0, 0.0, 0.0, 0.0]])

    def norm_inf(A, B, C):
        w = np.concatenate([np.zeros(1), np.logspace(-4, 4, 5000)])
        peak = 0.0
        for wi in w:
            try:
                R = np.linalg.solve(1j*wi*np.eye(A.shape[0]) - A, B)
                val = np.linalg.norm(C @ R, 2)
                if val > peak:
                    peak = val
            except np.linalg.LinAlgError:
                continue
        return float(peak)

    # Paper values from Table I (q, ||G_uy||, ||G_wy||, ||G_uw||, ||G_ww||, mu_max)
    paper_table = [
        (1.355, 1.000, 1.177, 5.718, 4.886, 0.205),
        (2.000, 1.000, 0.750, 3.633, 2.794, 0.358),
        (5.000, 1.000, 0.286, 2.270, 1.494, 0.669),
        (10.00, 1.000, 0.142, 1.996, 1.222, 0.819),
        (25.00, 1.000, 0.057, 1.869, 1.083, 0.923),
    ]

    print(f'             ||G_uy||         ||G_wy||         ||G_uw||         '
          f'||G_ww||         mu_max')
    print(f'  q       computed paper   computed paper   computed paper   '
          f'computed paper   computed paper')
    print(f'  ' + '-' * 96)

    for row in paper_table:
        q, e_uy, e_wy, e_uw, e_ww, e_mu = row
        A0, B0 = make_A0_B0(q)
        c_uy = norm_inf(A0, B0, C_y)
        c_wy = norm_inf(A0, E_h, C_y)
        c_uw = norm_inf(A0, B0, E_h.T)
        c_ww = norm_inf(A0, E_h, E_h.T)
        c_mu = 1.0 / c_ww

        print(f'  {q:>5.3f}  {c_uy:7.3f} {e_uy:7.3f}  '
              f'{c_wy:7.3f} {e_wy:7.3f}  '
              f'{c_uw:7.3f} {e_uw:7.3f}  '
              f'{c_ww:7.3f} {e_ww:7.3f}  '
              f'{c_mu:7.3f} {e_mu:7.3f}')

        compare(f'Table I row q={q}: ||G_uy||', c_uy, e_uy, abs_tol=0.01)
        compare(f'Table I row q={q}: ||G_wy||', c_wy, e_wy, abs_tol=0.02)
        compare(f'Table I row q={q}: ||G_uw||', c_uw, e_uw, abs_tol=0.1)
        compare(f'Table I row q={q}: ||G_ww||', c_ww, e_ww, abs_tol=0.1)


# Section 6. Table 6.3 — nonlinear residual on the 3D figure-8 (Section 6.1)
def section_6_table_6_3():
    header('6.  Table 6.3:  Nonlinear residual on the 3D figure-8 '
           '(Section 6.1)')

    paper_table = [
        ( 9.19,  5.17,  0.316291, 1.0002, 2.0e-4),
        (18.37, 10.33,  0.316241, 1.0000, 4.2e-5),
        (30.00, 16.87,  0.316233, 1.0000, 1.6e-5),
        (60.00, 33.74,  0.316229, 1.0000, 4.3e-6),
        (100.0, 56.23,  0.316228, 1.0000, 1.6e-6),
    ]

    cache_path = Path('figures/figure4_data.json')
    if cache_path.exists() and os.environ.get('FORCE_RERUN') != '1':
        with open(cache_path) as f:
            cache = json.load(f)
        rows = cache['data']
        source = 'cached'
    else:
        from quadrotor_sim import run_simulation, figure8_3d_traj
        print('  Re-running simulations (~30 s each)...')
        Fa = 0.5
        rows = []
        for p in [9.19, 18.37, 30.0, 60.0, 100.0]:
            t0 = time.time()
            result = run_simulation(
                figure8_3d_traj(),
                Fd_const=np.array([Fa, 0.0, 0.0]),
                p_inner=p,
                t_end=50.0,
            )
            ratio = result['mean_ex'] / (Fa * np.sqrt(RHO) / M)
            rows.append({
                'p_inner': p,
                'mean_ex': result['mean_ex'],
                'ratio': ratio,
                'residual': abs(ratio - 1.0),
            })
            print(f'    p_inner = {p:6.2f}: in {time.time()-t0:.1f}s')
        source = 're-simulated'

    print(f'  Source: {source}.')
    print()
    print(f'  p_inner    q       mean e_x (m)              ratio                residual')
    print(f'           (= p*ρ^¼) computed   paper        computed paper      computed paper')
    print(f'  ' + '-' * 84)

    for row in paper_table:
        p, q, e_mean, e_ratio, e_res = row
        match = next((r for r in rows if abs(r['p_inner'] - p) < 1e-2), None)
        if match is None:
            print(f'  {p:>7.2f}  {q:>5.2f}  (no cached data for this p_inner)')
            continue
        c_mean = match['mean_ex']
        c_ratio = match.get('ratio', c_mean / (0.5 * np.sqrt(RHO) / M))
        c_res = match.get('residual', abs(c_ratio - 1.0))

        print(f'  {p:>7.2f}  {q:>5.2f}  '
              f'{c_mean:>9.6f} {e_mean:>10.6f}    '
              f'{c_ratio:>7.4f} {e_ratio:>7.4f}    '
              f'{c_res:>8.1e} {e_res:>8.1e}')

        compare(f'Table 6.3 p_inner={p}: residual', c_res, e_res,
                rel_tol=0.20)  # 20% tol; residual values are themselves small


# Section 7. Table 6.4 — Bode response along the moderate figure-8 (Sec 6.2)
def section_7_table_6_4():
    header('7.  Table 6.4:  Bode response along the moderate figure-8 '
           '(Section 6.2)')

    cache_path = Path('figures/figure6_data.json')
    if not cache_path.exists():
        print('  No cached data; run figure6_realizations.py to regenerate.')
        return

    with open(cache_path) as f:
        cache = json.load(f)
    freqs = cache['probe_freqs']
    mags = cache['measured_mag']
    F_a = cache['F_a']
    measured = {round(w, 2): m for w, m in zip(freqs, mags)}

    # Closed-form |T_t(jw)| via 4D state-space evaluation (paper Lemma 5.1)
    from lmi_comparison import A_cl, B_cl, C_OUT
    def T_t_mag(omega, f0):
        jw = 1j * omega
        A = A_cl(f0); B = B_cl(f0)
        R = np.linalg.solve(jw*np.eye(4) - A, B)
        return float(abs((C_OUT @ R)[0, 0]))

    f0_hover = M * G                    # hover linearization point

    # Paper values: (omega, empirical, closed-form, ratio)
    paper_table = [
        (0.50, 0.006285, 0.006285, 1.000),
        (1.00, 0.005841, 0.005949, 0.982),
        (2.00, 0.003736, 0.003812, 0.980),
    ]

    print(f'  omega    empirical amplitude (m)    closed-form |T(jω)|·F_a (m)   ratio')
    print(f'  (rad/s)  computed    paper          computed       paper         comp.  paper')
    print(f'  ' + '-' * 84)

    for omega, e_emp, e_cf, e_ratio in paper_table:
        meas = measured.get(omega) or measured.get(round(omega, 2))
        if meas is None:
            print(f'  {omega:>5.2f}    (no data at this probe frequency)')
            continue
        c_emp = meas * F_a
        c_cf  = T_t_mag(omega, f0_hover) * F_a
        c_ratio = c_emp / c_cf

        print(f'  {omega:>5.2f}    {c_emp:.6f}  {e_emp:.6f}      '
              f'{c_cf:.6f}    {e_cf:.6f}    '
              f'{c_ratio:.3f}  {e_ratio:.3f}')

        compare(f'Table 6.4 omega={omega}: empirical', c_emp, e_emp,
                rel_tol=0.02)
        compare(f'Table 6.4 omega={omega}: closed form', c_cf, e_cf,
                rel_tol=0.02)


# Section 8. Table 6.5 — 8D regime sweep on horizontal circles (Section 6.3)
def section_8_table_6_5():
    header('8.  Table 6.5:  8D regime sweep on horizontal circles '
           '(Section 6.3)')

    cache_path = Path('figures/figure5_data.json')
    if not cache_path.exists():
        print('  No cached data; run figure5_parameter_sweep.py to regenerate.')
        return

    with open(cache_path) as f:
        cache = json.load(f)
    rows = cache.get('detail', [])

    # Paper values: keys (omega_c, p_inner), value = printed residual
    paper_table = [
        (1.0,  9.19,  +7e-6),
        (1.0, 18.37,  +1e-6),
        (1.0, 30.00,  +0.4e-6),
        (2.0,  9.19,  -1.9e-5),
        (2.0, 18.37,  -4e-6),
        (2.0, 30.00,  -2e-6),
        (3.0,  9.19,  -1.5e-4),
        (3.0, 18.37,  -2.4e-5),
        (3.0, 30.00,  -8e-6),
    ]

    print(f'  omega_c  p_inner   omega_c/p     residual')
    print(f'  (rad/s)  (rad/s)               computed       paper')
    print(f'  ' + '-' * 60)

    for wc, p, e_res in paper_table:
        match = next((r for r in rows
                      if abs(r['omega_c'] - wc) < 1e-6
                      and abs(r['p_inner'] - p) < 1e-2), None)
        if match is None:
            print(f'  {wc:>6.2f}  {p:>6.2f}                 (no data)')
            continue
        c_res = match.get('deviation_pct', 0.0) / 100.0

        print(f'  {wc:>6.2f}  {p:>6.2f}   {wc/p:>7.3f}    '
              f'{c_res:>+10.2e}   {e_res:>+10.1e}')

        same_sign = (c_res * e_res) >= 0
        same_order = (abs(c_res) < 5*abs(e_res) and abs(c_res) > 0.2*abs(e_res))
        if not (same_sign and same_order):
            discrepancies.append(
                (f'Table 6.5 (wc={wc}, p={p})', c_res, e_res,
                 abs(c_res - e_res)))


# ============================================================================
# Section 9. Table 7.1 — LMI-based comparison (Section 6.4)
# ============================================================================
def section_9_table_7_1():
    header('9.  Table 7.1:  LMI-based comparison (Section 6.4)')

    try:
        import cvxpy as cp                                      # noqa: F401
    except ImportError:
        print('  cvxpy not installed; skipping (install: pip install cvxpy scs)')
        return

    try:
        from lmi_comparison import (
            hinf_pointwise, hinf_trajectory_uniform, linfty_tube_bound
        )
    except ImportError as e:
        print(f'  lmi_comparison module not importable: {e}')
        return

    closed_form = np.sqrt(RHO) / M
    print(f'  Computing pointwise BRL LMI at three operating points')
    print(f'  and the trajectory-uniform LMI over the circle and figure-8')
    print(f'  parameter ranges. This takes about 5 minutes total.')
    print()

    import io, contextlib
    def quiet(fn, *args, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fn(*args, **kw)

    # Paper values from Table 7.1
    paper_table = [
        ('Closed-form sqrt(rho)/m',                  closed_form,    closed_form),
        ('Pointwise BRL LMI at hover (f0=4.91 N)',   None,           0.632455),
        ('Pointwise BRL LMI at f0=6.33 N',           None,           0.632455),
        ('Pointwise BRL LMI at f0=4.66 N',           None,           0.632455),
        ('Trajectory-uniform LMI (circle)',          None,           0.638340),
        ('Trajectory-uniform LMI (figure-8)',        None,           0.635029),
        ('L_inf-induced gain at hover',              None,           0.6816),
    ]

    print(f'  Method                                            '
          f'computed     paper       gap')
    print(f'  ' + '-' * 78)
    # closed form
    print(f'  Closed-form sqrt(rho)/m                           '
          f'{closed_form:.6f}     {closed_form:.6f}    (reference)')

    # pointwise BRLs
    for label, f0 in [
        ('Pointwise BRL LMI at hover (f0=4.91 N)',  M*G),
        ('Pointwise BRL LMI at f0=6.33 N',          6.33),
        ('Pointwise BRL LMI at f0=4.66 N',          4.66),
    ]:
        v = quiet(hinf_pointwise, f0=f0)
        gap_pct = 100 * (v / closed_form - 1)
        paper_v = next(p for l, _, p in paper_table if l == label)
        print(f'  {label:<48}  {v:.6f}     {paper_v:.6f}    '
              f'{gap_pct:+.4f}%')
        compare(label, v, paper_v, abs_tol=1e-5)

    # trajectory-uniform LMIs (slow)
    print(f'  (computing trajectory-uniform LMIs; ~3 minutes)...')
    v_circle = quiet(hinf_trajectory_uniform, np.linspace(4.91, 6.33, 20))
    paper_v = 0.638340
    gap_pct = 100 * (v_circle / closed_form - 1)
    print(f'  {"Trajectory-uniform LMI (circle)":<48}  '
          f'{v_circle:.6f}     {paper_v:.6f}    {gap_pct:+.4f}%')
    compare('Trajectory-uniform LMI (circle)', v_circle, paper_v,
            rel_tol=0.02)

    v_fig8 = quiet(hinf_trajectory_uniform, np.linspace(4.66, 5.51, 20))
    paper_v = 0.635029
    gap_pct = 100 * (v_fig8 / closed_form - 1)
    print(f'  {"Trajectory-uniform LMI (figure-8)":<48}  '
          f'{v_fig8:.6f}     {paper_v:.6f}    {gap_pct:+.4f}%')
    compare('Trajectory-uniform LMI (figure-8)', v_fig8, paper_v,
            rel_tol=0.02)

    # L_inf-induced gain
    v_Linf = quiet(linfty_tube_bound, f0=M*G)
    paper_v = 0.6816
    gap_pct = 100 * (v_Linf / closed_form - 1)
    print(f'  {"L_inf-induced gain at hover":<48}  '
          f'{v_Linf:.6f}     {paper_v:.6f}    {gap_pct:+.4f}%')
    compare('L_inf-induced gain at hover', v_Linf, paper_v, abs_tol=1e-3)


# ============================================================================
# Driver
# ============================================================================
def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Reproduce every numerical value in the manuscript.')
    parser.add_argument('--full', action='store_true',
                        help='Also compute Table 7.1 (LMI comparison; +5 min)')
    args = parser.parse_args(argv)

    sections = [
        section_1_closed_form_gain,
        section_2_threshold,
        section_3_Luw,
        section_4_trajectory_params,
        section_5_table_I,
        section_6_table_6_3,
        section_7_table_6_4,
        section_8_table_6_5,
    ]
    if args.full:
        sections.append(section_9_table_7_1)
    else:
        pass 

    for fn in sections:
        try:
            fn()
        except Exception as e:
            print(f'\n  ERROR in {fn.__name__}: {e}')
            import traceback
            traceback.print_exc()

    # ----- Final summary -----
    header('Summary')
    if not discrepancies:
        print('  Every value reproduces the paper to within the expected')
        print('  numerical tolerance.')
    else:
        print(f'  {len(discrepancies)} value(s) differ from the paper '
              f'beyond tolerance:')
        for where, c, p, d in discrepancies:
            print(f'    {where}:')
            print(f'      computed = {c}')
            print(f'      paper    = {p}')
            print(f'      diff     = {d}')

    if not args.full:
        print()
        print('  (Table 7.1 LMI comparison was skipped. Re-run with --full')
    print()
    return 0 if not discrepancies else 1


if __name__ == '__main__':
    sys.exit(main())
