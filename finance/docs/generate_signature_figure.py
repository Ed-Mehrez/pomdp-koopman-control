"""
Generate beautiful signature visualization figures for the presentation.
Shows how signatures capture path geometry through iterated integrals.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PolyCollection

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


def generate_single_path_signatures():
    """
    Show how signature features EVOLVE over time on ONE path.
    This is what we actually observe in economics - one realization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    np.random.seed(42)
    n = 300
    dt = 1.0 / n
    t = np.linspace(0, 1, n)

    # Simulate a realistic price path with regime change
    # Regime 1: Low vol (t < 0.4)
    # Regime 2: High vol crisis (0.4 < t < 0.6)
    # Regime 3: Recovery (t > 0.6)
    vol = np.where(t < 0.4, 0.1,
                   np.where(t < 0.6, 0.3, 0.15))
    drift = np.where(t < 0.4, 0.2,
                     np.where(t < 0.6, -0.5, 0.3))

    increments = drift * dt + vol * np.sqrt(dt) * np.random.randn(n)
    path = np.cumsum(increments)

    colors = {'path': '#2c3e50', 'sig1': '#27ae60', 'qv': '#e74c3c', 'area': '#3498db'}

    # Panel 1: The price path with regime shading
    ax1 = axes[0, 0]
    ax1.plot(t, path, color=colors['path'], linewidth=2)
    ax1.axvspan(0, 0.4, alpha=0.1, color='green', label='Low Vol')
    ax1.axvspan(0.4, 0.6, alpha=0.1, color='red', label='Crisis')
    ax1.axvspan(0.6, 1, alpha=0.1, color='blue', label='Recovery')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_title('ONE Price Path (What We Observe)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative signature Level 1 (total return so far)
    ax2 = axes[0, 1]
    cumsum_return = np.cumsum(increments)
    ax2.plot(t, cumsum_return, color=colors['sig1'], linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvspan(0.4, 0.6, alpha=0.1, color='red')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Cumulative Return', fontsize=11)
    ax2.set_title('Level 1: $S^{(1)}_t = X_t - X_0$\n(Running total return)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Running quadratic variation (Level 2 diagonal)
    ax3 = axes[1, 0]
    running_qv = np.cumsum(increments**2)
    ax3.plot(t, running_qv, color=colors['qv'], linewidth=2)
    ax3.axvspan(0.4, 0.6, alpha=0.1, color='red')

    # Add slope annotations
    slope1 = (running_qv[int(0.4*n)] - running_qv[0]) / 0.4
    slope2 = (running_qv[int(0.6*n)] - running_qv[int(0.4*n)]) / 0.2
    ax3.annotate(f'Slope ≈ {slope1:.3f}\n(Low vol)', xy=(0.2, running_qv[int(0.2*n)]),
                 fontsize=9, color='green')
    ax3.annotate(f'Slope ≈ {slope2:.3f}\n(High vol!)', xy=(0.5, running_qv[int(0.5*n)]),
                 fontsize=9, color='red')

    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Cumulative QV', fontsize=11)
    ax3.set_title('Level 2: $S^{(2)}_{XX,t} = \\sum_{s<t} (\\Delta X_s)^2$\n(Running volatility)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel 4: What signatures reveal vs raw data
    ax4 = axes[1, 1]

    # Compare: Raw returns vs signature-derived volatility estimate
    window = 20
    raw_vol = np.array([np.std(increments[max(0,i-window):i+1]) if i > 0 else 0 for i in range(n)])
    sig_vol = np.sqrt(np.diff(running_qv, prepend=0) / dt)  # Instantaneous from signature

    # Smooth for visualization
    from scipy.ndimage import uniform_filter1d
    sig_vol_smooth = uniform_filter1d(sig_vol, size=window)

    ax4.plot(t, vol, 'k--', linewidth=2, label='True Vol (hidden)', alpha=0.7)
    ax4.plot(t, sig_vol_smooth, color=colors['qv'], linewidth=2, label='From Signature QV')
    ax4.axvspan(0.4, 0.6, alpha=0.1, color='red')

    ax4.set_xlabel('Time', fontsize=11)
    ax4.set_ylabel('Volatility', fontsize=11)
    ax4.set_title('Signatures Reveal Hidden State\nfrom ONE Path!', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 0.5)

    plt.suptitle('Extracting Information from a Single Realization via Signatures',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('single_path_signatures.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved single_path_signatures.png")


def generate_signature_geometry_figure():
    """
    Create a figure showing WHAT signatures compute geometrically.
    - Show a simple 2D path
    - Shade the areas corresponding to signature terms
    - Show that S^{AB} ≠ S^{BA} (order matters)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Simple discrete path: 3 points forming a triangle-ish shape
    # Time-augmented: (t, X) where t = [0, 1, 2] and X = [0, 2, 1]

    # Panel 1: Level 1 - just endpoints
    ax1 = axes[0]
    t = np.array([0, 1, 2])
    X = np.array([0, 2, 1])

    ax1.plot(t, X, 'o-', color='#2c3e50', linewidth=2.5, markersize=10)
    ax1.scatter([0, 2], [0, 1], s=150, c=['green', 'red'], zorder=5)

    # Arrow showing total displacement
    ax1.annotate('', xy=(2, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
    ax1.text(1.3, 0.2, '$S^{(1)} = X_T - X_0 = 1$', fontsize=14, color='#27ae60',
            fontweight='bold')

    ax1.set_xlabel('Time $t$', fontsize=12)
    ax1.set_ylabel('Price $X$', fontsize=12)
    ax1.set_title('Level 1: Total Displacement\n"Where did we end up?"', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.3, 2.5)
    ax1.set_ylim(-0.5, 2.5)

    # Panel 2: Level 2 diagonal - sum of squared increments
    ax2 = axes[1]
    ax2.plot(t, X, 'o-', color='#2c3e50', linewidth=2.5, markersize=10)

    # Show the increments with bars
    dX = np.diff(X)  # [2, -1]
    for i, dx in enumerate(dX):
        # Draw a square showing dx²
        sq_size = abs(dx) * 0.3
        rect = mpatches.Rectangle((t[i] + 0.5 - sq_size/2, -0.3),
                                   sq_size, sq_size,
                                   facecolor='#e74c3c', alpha=0.5, edgecolor='#c0392b', lw=2)
        ax2.add_patch(rect)
        ax2.text(t[i] + 0.5, -0.3 + sq_size/2, f'${dx}^2={dx**2}$',
                ha='center', va='center', fontsize=10, fontweight='bold')

    ax2.text(1, 2.2, f'$S^{{(2)}} = \\sum (\\Delta X)^2 = {dX[0]}^2 + {dX[1]}^2 = {np.sum(dX**2)}$',
            fontsize=12, ha='center', color='#e74c3c', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Time $t$', fontsize=12)
    ax2.set_ylabel('Price $X$', fontsize=12)
    ax2.set_title('Level 2 (Diagonal): Quadratic Variation\n"How much did we wiggle?"', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.3, 2.5)
    ax2.set_ylim(-0.8, 2.5)

    # Panel 3: Level 2 cross-terms - the key insight about order
    ax3 = axes[2]

    # Now show a 2D path: Asset A vs Asset B
    # Path: (0,0) -> (2,1) -> (1,2) (A rises first, then B rises while A falls)
    A = np.array([0, 2, 1])
    B = np.array([0, 1, 2])

    ax3.plot(A, B, 'o-', color='#2c3e50', linewidth=2.5, markersize=10)
    ax3.scatter([A[0], A[-1]], [B[0], B[-1]], s=150, c=['green', 'red'], zorder=5)

    # Show direction with arrows
    for i in range(len(A)-1):
        ax3.annotate('', xy=(A[i+1], B[i+1]), xytext=(A[i], B[i]),
                    arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))

    # Shade the area - this is the Lévy area
    # The signed area can be computed as the shoelace formula
    # Fill the region enclosed by the path and the line from end to start
    from matplotlib.patches import Polygon
    # Close the path
    polygon_pts = np.column_stack([np.append(A, A[0]), np.append(B, B[0])])
    poly = Polygon(polygon_pts[:-1], alpha=0.3, facecolor='#9b59b6', edgecolor='#8e44ad', lw=2)
    ax3.add_patch(poly)

    # Compute Lévy area
    dA = np.diff(A)
    dB = np.diff(B)
    S_AB = np.sum(A[:-1] * dB)  # ∫ A dB
    S_BA = np.sum(B[:-1] * dA)  # ∫ B dA
    levy = 0.5 * (S_AB - S_BA)

    ax3.text(0.8, 1.5, f'Lévy Area\n$= \\frac{{1}}{{2}}(S^{{AB}} - S^{{BA}})$\n$= {levy:.1f}$',
            fontsize=11, ha='center', color='#8e44ad', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Add computation details
    ax3.text(0.5, -0.4, f'$S^{{AB}} = \\sum A_i \\Delta B_i = {S_AB}$\n$S^{{BA}} = \\sum B_i \\Delta A_i = {S_BA}$',
            fontsize=10, ha='left', transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))

    ax3.set_xlabel('Asset A', fontsize=12)
    ax3.set_ylabel('Asset B', fontsize=12)
    ax3.set_title('Level 2 (Cross): Lévy Area\n"Who moved first?" (Order matters!)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.5, 2.5)
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_aspect('equal')

    plt.suptitle('What Signatures Actually Compute (Geometric View)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('signature_geometry.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved signature_geometry.png")


def generate_signature_levels_figure():
    """
    Create a SIMPLE figure showing signature levels on ONE path.
    Focus on intuition, not complexity.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    np.random.seed(42)
    n = 150
    t = np.linspace(0, 1, n)

    # A realistic price path with volatility clustering
    # High vol at start, low vol in middle, high vol at end
    vol = 0.1 + 0.15 * np.exp(-((t - 0.2)**2) / 0.01) + 0.12 * np.exp(-((t - 0.8)**2) / 0.02)
    increments = vol * np.random.randn(n) * np.sqrt(1/n)
    path = np.cumsum(increments)
    path = path - path[0]  # Start at 0

    colors = {'path': '#2c3e50', 'l1': '#27ae60', 'l2': '#e74c3c'}

    # Panel 1: Level 1 = Where did we end up?
    ax1 = axes[0]
    ax1.plot(t, path, color=colors['path'], linewidth=2)
    ax1.scatter([0, 1], [path[0], path[-1]], s=120, c=['green', 'red'], zorder=5)

    # Arrow showing total displacement
    ax1.annotate('', xy=(1.05, path[-1]), xytext=(1.05, path[0]),
                 arrowprops=dict(arrowstyle='<->', color=colors['l1'], lw=3))
    ax1.text(1.08, (path[0] + path[-1])/2, f'$\\Delta X = {path[-1]:.2f}$',
             fontsize=12, color=colors['l1'], fontweight='bold', va='center')

    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_title('Level 1: Total Return\n"Where did we end up?"', fontsize=13, fontweight='bold')
    ax1.set_xlim(-0.05, 1.2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Level 2 (Diagonal) = How bumpy was the ride?
    ax2 = axes[1]
    ax2.plot(t, path, color=colors['path'], linewidth=2, alpha=0.5)

    # Show squared increments as bars
    sq_inc = increments**2
    # Normalize for visualization
    sq_inc_scaled = sq_inc / np.max(sq_inc) * 0.3 * (np.max(path) - np.min(path))
    for i in range(0, n-1, 3):  # Every 3rd point for clarity
        ax2.bar(t[i], sq_inc_scaled[i], width=0.02, bottom=np.min(path)-0.1,
                color=colors['l2'], alpha=0.6, edgecolor='none')

    qv = np.sum(sq_inc)
    ax2.text(0.5, 0.85, f'QV = $\\sum (\\Delta X)^2 = {qv:.3f}$',
             transform=ax2.transAxes, fontsize=12, color=colors['l2'],
             fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Price', fontsize=11)
    ax2.set_title('Level 2: Quadratic Variation\n"How bumpy was the ride?"', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Higher Levels = Path Shape Details
    ax3 = axes[2]

    # Show two paths with SAME Level 1 and Level 2, but different higher levels
    np.random.seed(100)
    # Path A: Smooth rise then plateau
    path_a = 0.3 * (1 - np.exp(-5*t))
    # Path B: Oscillate then rise
    path_b = 0.15 * np.sin(4*np.pi*t) * (1-t) + 0.3 * t

    # Normalize to same endpoint and same QV (approximately)
    path_b = path_b * (path_a[-1] / path_b[-1])

    ax3.plot(t, path_a, color='#3498db', linewidth=2.5, label='Path A: Early rise')
    ax3.plot(t, path_b, color='#9b59b6', linewidth=2.5, label='Path B: Oscillate')
    ax3.scatter([0, 1, 0, 1], [path_a[0], path_a[-1], path_b[0], path_b[-1]],
                s=80, c=['green', 'red', 'green', 'red'], zorder=5)

    ax3.text(0.5, 0.15, 'Same endpoint,\ndifferent shape\n→ Different higher signatures',
             transform=ax3.transAxes, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Price', fontsize=11)
    ax3.set_title('Higher Levels: Path Shape\n"How did we get there?"', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Signature Levels Capture Progressively Finer Path Information',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('signature_levels_explained.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved signature_levels_explained.png")


def generate_method_comparison_figure():
    """
    Create a clean comparison of estimation methods.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Data from experiments
    methods = ['Oracle BPF\n(True Params)', 'Rolling BV\n(Non-parametric)',
               'Sig-KKF\n(Model-Free)', 'Operator Method\n(Pure Geometry)']
    mse_values = [1.1e-4, 2.7e-4, 1.3e-3, 8.9e-4]
    relative = [1.0, 2.5, 11.8, 8.1]

    colors = ['#27ae60', '#3498db', '#e74c3c', '#f39c12']

    bars = ax.bar(methods, mse_values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val, rel in zip(bars, mse_values, relative):
        height = bar.get_height()
        ax.annotate(f'MSE: {val:.1e}\n({rel:.1f}× Oracle)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Bates Model: Volatility Estimation Comparison\n(Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(5e-5, 5e-3)
    ax.grid(True, alpha=0.3, axis='y')

    # Add a horizontal line at Oracle level
    ax.axhline(mse_values[0], color='#27ae60', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(3.5, mse_values[0] * 1.2, 'Oracle Baseline', fontsize=10, color='#27ae60')

    plt.tight_layout()
    plt.savefig('method_comparison_bar.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved method_comparison_bar.png")


def generate_pipeline_figure():
    """
    Create a visual pipeline diagram showing the Sig-KKF workflow.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Box positions and sizes
    boxes = [
        (0.5, 1.5, 2.5, 1.5, 'Price Path\n$S_{[0,t]}$', '#ecf0f1'),
        (3.5, 1.5, 2.5, 1.5, 'Signature\n$\\Psi(S) = [S^{(1)}, S^{(2)}, ...]$', '#3498db'),
        (6.5, 1.5, 2.5, 1.5, 'Koopman\nOperator $K$', '#9b59b6'),
        (9.5, 1.5, 2.5, 1.5, 'Linear\nKalman Filter', '#e74c3c'),
        (12.5, 1.5, 1.3, 1.5, '$\\hat{v}_t$', '#27ae60'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.05,rounding_size=0.2",
                                        facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white' if color not in ['#ecf0f1'] else 'black')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=2)
    arrows = [(3.0, 2.25), (6.0, 2.25), (9.0, 2.25), (12.0, 2.25)]
    for x, y in arrows:
        ax.annotate('', xy=(x+0.4, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

    # Labels below
    labels = [
        (1.75, 0.8, 'Lift to\nFeature Space'),
        (4.75, 0.8, 'Learn Linear\nDynamics'),
        (7.75, 0.8, 'Bayesian\nUpdate'),
        (10.75, 0.8, 'Project to\nObservable'),
    ]
    for x, y, text in labels:
        ax.text(x, y, text, ha='center', va='top', fontsize=9, style='italic')

    ax.set_title('Sig-KKF Pipeline: Nonlinear Filtering via Linear Algebra',
                 fontsize=14, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.savefig('sigkkf_pipeline.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved sigkkf_pipeline.png")


def generate_why_signatures_matter():
    """
    Create a figure comparing cross-correlation vs Lévy Area.
    Be HONEST: cross-correlation CAN detect simple lead-lag.
    Show WHERE signatures add value.
    """
    fig = plt.figure(figsize=(15, 10))

    # Layout: 3 rows
    # Row 1: The math formulas
    # Row 2: Simple lead-lag (BOTH methods work)
    # Row 3: Complex/nonlinear case (signatures help)

    np.random.seed(42)
    n = 200
    t = np.linspace(0, 1, n)

    # =========================================================================
    # ROW 1: Math comparison (text panel)
    # =========================================================================
    ax_math = fig.add_subplot(3, 1, 1)
    ax_math.axis('off')

    math_text = """
    CROSS-CORRELATION (what you know):                    LÉVY AREA (signature Level 2):

    $\\rho_{AB}(\\tau) = \\text{Corr}(A_t, B_{t+\\tau})$                   $\\mathcal{A} = \\frac{1}{2}\\int_0^T (A_t \\, dB_t - B_t \\, dA_t)$

    • Compute for each lag $\\tau$                              • Single number for entire path
    • Peak at $\\tau^* > 0$ means A leads B                     • $\\mathcal{A} > 0$ means B leads A (counterclockwise)
    • Requires choosing which lags to check                 • No lag parameter needed
    • Linear dependence only                                • Captures path geometry

    BOTH detect simple lead-lag! Signatures add value for: nonlinear dynamics, variable lags, integration with Koopman.
    """
    ax_math.text(0.5, 0.5, math_text, transform=ax_math.transAxes,
                fontsize=11, family='monospace', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#bdc3c7'))

    # =========================================================================
    # ROW 2: Simple linear lead-lag - BOTH methods work
    # =========================================================================
    # Create a simple lagged relationship
    lag = 30  # B follows A with this lag
    noise_A = np.cumsum(np.random.randn(n + lag)) * 0.1
    A_simple = noise_A[lag:]  # A starts at index 'lag'
    B_simple = noise_A[:-lag]  # B is lagged version

    # Normalize
    A_simple = (A_simple - A_simple.mean()) / A_simple.std()
    B_simple = (B_simple - B_simple.mean()) / B_simple.std()

    ax_ts1 = fig.add_subplot(3, 3, 4)
    ax_ts1.plot(t, A_simple, color='#3498db', linewidth=2, label='A')
    ax_ts1.plot(t, B_simple, color='#e67e22', linewidth=2, linestyle='--', label='B')
    ax_ts1.set_title('Simple Lead-Lag\n(A leads B by fixed lag)', fontweight='bold')
    ax_ts1.set_xlabel('Time')
    ax_ts1.set_ylabel('Value')
    ax_ts1.legend(fontsize=9)
    ax_ts1.grid(True, alpha=0.3)

    # Cross-correlation
    ax_xcorr1 = fig.add_subplot(3, 3, 5)
    max_lag = 60
    lags = np.arange(-max_lag, max_lag + 1)
    xcorr = np.correlate(A_simple - A_simple.mean(), B_simple - B_simple.mean(), mode='full')
    xcorr = xcorr[n - max_lag - 1: n + max_lag] / (n * A_simple.std() * B_simple.std())
    ax_xcorr1.plot(lags, xcorr, color='#9b59b6', linewidth=2)
    peak_lag = lags[np.argmax(xcorr)]
    ax_xcorr1.axvline(peak_lag, color='red', linestyle='--', label=f'Peak at τ={peak_lag}')
    ax_xcorr1.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax_xcorr1.set_title(f'Cross-Correlation\n✓ Detects lead (peak at τ={peak_lag})', fontweight='bold')
    ax_xcorr1.set_xlabel('Lag τ')
    ax_xcorr1.set_ylabel('Correlation')
    ax_xcorr1.legend(fontsize=9)
    ax_xcorr1.grid(True, alpha=0.3)

    # Lévy Area
    ax_levy1 = fig.add_subplot(3, 3, 6)
    da = np.diff(A_simple)
    db = np.diff(B_simple)
    levy = 0.5 * np.sum(A_simple[:-1] * db - B_simple[:-1] * da)
    # Plot path in (A,B) space
    ax_levy1.plot(A_simple, B_simple, color='#27ae60', linewidth=1.5, alpha=0.7)
    ax_levy1.scatter([A_simple[0]], [B_simple[0]], s=80, c='green', zorder=5, marker='o')
    ax_levy1.scatter([A_simple[-1]], [B_simple[-1]], s=80, c='red', zorder=5, marker='s')
    sign = "+" if levy > 0 else "−"
    ax_levy1.set_title(f'Lévy Area = {sign}{abs(levy):.2f}\n✓ Also detects lead!', fontweight='bold')
    ax_levy1.set_xlabel('A')
    ax_levy1.set_ylabel('B')
    ax_levy1.grid(True, alpha=0.3)
    ax_levy1.set_aspect('equal')

    # =========================================================================
    # ROW 3: Complex case - variable lag / nonlinear
    # =========================================================================
    # B responds to A with VARIABLE lag (fast when A jumps, slow otherwise)
    np.random.seed(123)
    A_complex = np.zeros(n)
    B_complex = np.zeros(n)

    # A has regime changes
    A_complex[:50] = np.linspace(0, 0.5, 50)
    A_complex[50:80] = 0.5 + 0.3 * np.sin(np.linspace(0, 2*np.pi, 30))  # oscillation
    A_complex[80:120] = np.linspace(A_complex[79], 1.5, 40)  # jump up
    A_complex[120:] = 1.5 + 0.1 * np.cumsum(np.random.randn(80)) / np.sqrt(80)

    # B responds with VARIABLE lag - fast for big moves, slow for small
    for i in range(1, n):
        # Adaptive lag: respond faster to big changes in A
        dA = abs(A_complex[i] - A_complex[i-1])
        response_speed = 0.05 + 0.3 * min(dA * 10, 1)  # faster for big moves
        B_complex[i] = B_complex[i-1] + response_speed * (A_complex[i-1] - B_complex[i-1])

    # Normalize
    A_complex = (A_complex - A_complex.mean()) / (A_complex.std() + 1e-9)
    B_complex = (B_complex - B_complex.mean()) / (B_complex.std() + 1e-9)

    ax_ts2 = fig.add_subplot(3, 3, 7)
    ax_ts2.plot(t, A_complex, color='#3498db', linewidth=2, label='A')
    ax_ts2.plot(t, B_complex, color='#e67e22', linewidth=2, linestyle='--', label='B')
    ax_ts2.set_title('Complex: Variable Lag\n(B responds faster to big moves)', fontweight='bold')
    ax_ts2.set_xlabel('Time')
    ax_ts2.set_ylabel('Value')
    ax_ts2.legend(fontsize=9)
    ax_ts2.grid(True, alpha=0.3)

    # Cross-correlation - now ambiguous
    ax_xcorr2 = fig.add_subplot(3, 3, 8)
    xcorr2 = np.correlate(A_complex - A_complex.mean(), B_complex - B_complex.mean(), mode='full')
    xcorr2 = xcorr2[n - max_lag - 1: n + max_lag] / (n * A_complex.std() * B_complex.std())
    ax_xcorr2.plot(lags, xcorr2, color='#9b59b6', linewidth=2)
    peak_lag2 = lags[np.argmax(xcorr2)]
    ax_xcorr2.axvline(peak_lag2, color='red', linestyle='--', alpha=0.5)
    ax_xcorr2.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax_xcorr2.set_title(f'Cross-Correlation\n⚠ Peak at τ={peak_lag2}, but misleading!', fontweight='bold')
    ax_xcorr2.set_xlabel('Lag τ')
    ax_xcorr2.set_ylabel('Correlation')
    ax_xcorr2.grid(True, alpha=0.3)
    ax_xcorr2.annotate('Lag varies\nover time!', xy=(peak_lag2, xcorr2[max_lag + peak_lag2]),
                       xytext=(peak_lag2 + 15, xcorr2[max_lag + peak_lag2] + 0.1),
                       fontsize=9, color='red',
                       arrowprops=dict(arrowstyle='->', color='red'))

    # Lévy Area - still captures the asymmetry
    ax_levy2 = fig.add_subplot(3, 3, 9)
    da2 = np.diff(A_complex)
    db2 = np.diff(B_complex)
    levy2 = 0.5 * np.sum(A_complex[:-1] * db2 - B_complex[:-1] * da2)
    ax_levy2.plot(A_complex, B_complex, color='#27ae60', linewidth=1.5, alpha=0.7)
    ax_levy2.scatter([A_complex[0]], [B_complex[0]], s=80, c='green', zorder=5, marker='o')
    ax_levy2.scatter([A_complex[-1]], [B_complex[-1]], s=80, c='red', zorder=5, marker='s')
    sign2 = "+" if levy2 > 0 else "−"
    ax_levy2.set_title(f'Lévy Area = {sign2}{abs(levy2):.2f}\n✓ Captures total lead-lag', fontweight='bold')
    ax_levy2.set_xlabel('A')
    ax_levy2.set_ylabel('B')
    ax_levy2.grid(True, alpha=0.3)
    ax_levy2.set_aspect('equal')

    plt.suptitle('Cross-Correlation vs Lévy Area: When Does Each Work?',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('why_signatures_matter.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved why_signatures_matter.png")


def generate_distribution_fingerprint():
    """
    Show that different stochastic processes have different signature distributions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    np.random.seed(42)
    n_paths = 200
    n_steps = 50

    def simulate_gbm(n_paths, n_steps, sigma=0.2):
        """Geometric Brownian Motion - independent increments."""
        paths = np.zeros((n_paths, n_steps))
        for i in range(n_paths):
            W = np.random.randn(n_steps) * np.sqrt(1/n_steps)
            paths[i] = np.cumsum(sigma * W)
        return paths

    def simulate_ou(n_paths, n_steps, theta=5.0, sigma=0.3):
        """Ornstein-Uhlenbeck - mean-reverting."""
        paths = np.zeros((n_paths, n_steps))
        dt = 1/n_steps
        for i in range(n_paths):
            x = 0
            for t in range(n_steps):
                x = x - theta * x * dt + sigma * np.sqrt(dt) * np.random.randn()
                paths[i, t] = x
        return paths

    def simulate_heston_vol(n_paths, n_steps, kappa=2.0, theta=0.04, xi=0.3):
        """Heston-like volatility clustering."""
        paths = np.zeros((n_paths, n_steps))
        dt = 1/n_steps
        for i in range(n_paths):
            v = theta
            x = 0
            for t in range(n_steps):
                v = max(v + kappa * (theta - v) * dt + xi * np.sqrt(max(v, 0) * dt) * np.random.randn(), 0.001)
                x = x + np.sqrt(v * dt) * np.random.randn()
                paths[i, t] = x
        return paths

    # Generate paths
    gbm_paths = simulate_gbm(n_paths, n_steps)
    ou_paths = simulate_ou(n_paths, n_steps)
    heston_paths = simulate_heston_vol(n_paths, n_steps)

    processes = [
        (gbm_paths, 'GBM (Random Walk)', '#3498db'),
        (ou_paths, 'OU (Mean-Reverting)', '#27ae60'),
        (heston_paths, 'Heston (Vol Clustering)', '#e74c3c')
    ]

    def compute_path_features(paths):
        """Compute simple signature-like features."""
        n = paths.shape[0]
        total_returns = paths[:, -1] - paths[:, 0]
        quad_vars = np.sum(np.diff(paths, axis=1)**2, axis=1)

        # Approximate "momentum" / trend strength
        mid = paths.shape[1] // 2
        first_half = paths[:, mid] - paths[:, 0]
        second_half = paths[:, -1] - paths[:, mid]
        asymmetry = first_half - second_half

        return total_returns, quad_vars, asymmetry

    # Top row: sample paths
    for i, (paths, name, color) in enumerate(processes):
        ax = axes[0, i]
        t = np.linspace(0, 1, n_steps)
        for j in range(min(30, n_paths)):
            ax.plot(t, paths[j], color=color, alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Bottom row: signature feature distributions
    all_features = [compute_path_features(p) for p, _, _ in processes]

    # Plot quadratic variation distributions
    ax_qv = axes[1, 0]
    for (_, qv, _), (_, name, color) in zip(all_features, processes):
        ax_qv.hist(qv, bins=25, alpha=0.5, color=color, label=name, density=True)
    ax_qv.set_xlabel('Quadratic Variation (Level 2 Sig)')
    ax_qv.set_ylabel('Density')
    ax_qv.set_title('Signature Level 2: Volatility', fontsize=12, fontweight='bold')
    ax_qv.legend(fontsize=8)
    ax_qv.grid(True, alpha=0.3)

    # Plot total return vs QV scatter
    ax_scatter = axes[1, 1]
    for (ret, qv, _), (_, name, color) in zip(all_features, processes):
        ax_scatter.scatter(ret, qv, alpha=0.4, color=color, label=name, s=20)
    ax_scatter.set_xlabel('Total Return (Level 1 Sig)')
    ax_scatter.set_ylabel('Quadratic Variation (Level 2 Sig)')
    ax_scatter.set_title('Signature Space: Different Processes Separate!', fontsize=12, fontweight='bold')
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(True, alpha=0.3)

    # Plot asymmetry
    ax_asym = axes[1, 2]
    for (_, _, asym), (_, name, color) in zip(all_features, processes):
        ax_asym.hist(asym, bins=25, alpha=0.5, color=color, label=name, density=True)
    ax_asym.set_xlabel('Path Asymmetry (Higher Level Feature)')
    ax_asym.set_ylabel('Density')
    ax_asym.set_title('Higher Signatures: Path Shape', fontsize=12, fontweight='bold')
    ax_asym.legend(fontsize=8)
    ax_asym.grid(True, alpha=0.3)

    plt.suptitle('Signatures as "Fingerprints" for Stochastic Processes\n'
                 'Different dynamics → Different signature distributions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('signature_fingerprints.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved signature_fingerprints.png")


def generate_information_preservation():
    """
    Show that signatures preserve enough information to distinguish economically
    meaningful scenarios.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    np.random.seed(42)

    # Scenario: Bull market vs Bear market paths
    n = 100
    t = np.linspace(0, 1, n)

    # Bull: steady upward with small corrections
    bull_drift = 0.5 * t
    bull_noise = 0.1 * np.cumsum(np.random.randn(n)) / np.sqrt(n)
    bull_path = bull_drift + bull_noise

    # Bear: sharp drops with dead cat bounces
    bear_base = -0.3 * t
    bear_noise = 0.15 * np.cumsum(np.random.randn(n)) / np.sqrt(n)
    # Add a crash and recovery pattern
    crash = -0.2 * np.exp(-((t - 0.4)**2) / 0.02)
    bounce = 0.1 * np.exp(-((t - 0.6)**2) / 0.01)
    bear_path = bear_base + bear_noise + crash + bounce

    # Left panel: The paths
    ax1 = axes[0]
    ax1.plot(t, bull_path, color='#27ae60', linewidth=2.5, label='Bull Market')
    ax1.plot(t, bear_path, color='#e74c3c', linewidth=2.5, label='Bear Market')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(t, 0, bull_path, where=bull_path > 0, alpha=0.2, color='#27ae60')
    ax1.fill_between(t, 0, bear_path, where=bear_path < 0, alpha=0.2, color='#e74c3c')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Cumulative Return', fontsize=11)
    ax1.set_title('Two Market Regimes', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: Signature features comparison
    ax2 = axes[1]

    def compute_features(path):
        """Compute signature-like features."""
        ret = path[-1] - path[0]
        qv = np.sum(np.diff(path)**2)
        # Max drawdown proxy
        running_max = np.maximum.accumulate(path)
        drawdown = np.max(running_max - path)
        # Trend consistency (how much time spent above starting point)
        time_positive = np.mean(path > path[0])
        return ret, qv, drawdown, time_positive

    bull_features = compute_features(bull_path)
    bear_features = compute_features(bear_path)

    feature_names = ['Total Return\n(Level 1)', 'Volatility\n(Level 2)',
                     'Max Drawdown\n(Path Feature)', 'Time Positive\n(Path Feature)']

    x = np.arange(len(feature_names))
    width = 0.35

    # Normalize for visualization
    bull_norm = np.array(bull_features)
    bear_norm = np.array(bear_features)

    # Scale to similar magnitudes for visualization
    scale = np.maximum(np.abs(bull_norm), np.abs(bear_norm)) + 0.01
    bull_scaled = bull_norm / scale
    bear_scaled = bear_norm / scale

    bars1 = ax2.bar(x - width/2, bull_scaled, width, label='Bull', color='#27ae60', edgecolor='black')
    bars2 = ax2.bar(x + width/2, bear_scaled, width, label='Bear', color='#e74c3c', edgecolor='black')

    ax2.set_ylabel('Normalized Feature Value', fontsize=11)
    ax2.set_title('Signature Features Clearly Separate Regimes', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add actual values as annotations
    for i, (b, r) in enumerate(zip(bull_features, bear_features)):
        ax2.annotate(f'{b:.2f}', (x[i] - width/2, bull_scaled[i]),
                     ha='center', va='bottom' if bull_scaled[i] > 0 else 'top',
                     fontsize=8, fontweight='bold')
        ax2.annotate(f'{r:.2f}', (x[i] + width/2, bear_scaled[i]),
                     ha='center', va='bottom' if bear_scaled[i] > 0 else 'top',
                     fontsize=8, fontweight='bold')

    plt.suptitle('Signatures Capture Economically Meaningful Information',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('signature_economic_meaning.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved signature_economic_meaning.png")


if __name__ == '__main__':
    print("Generating presentation figures...")
    generate_signature_geometry_figure()  # NEW: What signatures compute geometrically
    generate_single_path_signatures()  # For single-path signature evolution
    generate_signature_levels_figure()  # Clean 3-panel explanation
    generate_method_comparison_figure()
    generate_pipeline_figure()
    generate_why_signatures_matter()  # Comparison with cross-correlation
    generate_distribution_fingerprint()
    generate_information_preservation()
    print("\nAll figures generated successfully!")