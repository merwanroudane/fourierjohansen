"""
Comprehensive test script for fourier-johansen library.
Tests all main functions to verify correct operation.
"""

import numpy as np
import sys

print("=" * 60)
print("    FOURIER-JOHANSEN LIBRARY - FUNCTION VERIFICATION")
print("=" * 60)

# Generate test data
np.random.seed(42)
T = 150

# Create cointegrated data with smooth break
t = np.arange(T)
break_term = 3 * np.sin(2 * np.pi * t / T)

x1 = np.cumsum(np.random.randn(T)) + break_term
x2 = x1 + np.random.randn(T) * 0.5 + break_term * 0.8
x3 = np.cumsum(np.random.randn(T))
X = np.column_stack([x1, x2, x3])

print(f"\nTest data: T={T}, n={X.shape[1]} variables")
print("-" * 60)

# Test 1: Import all functions
print("\n[1] Testing imports...")
try:
    from fourier_johansen import (
        johansen, johansen_fourier, sc_vecm, sbc_test, union_test,
        to_latex, to_markdown
    )
    print("    [OK] All imports successful")
except Exception as e:
    print(f"    [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Standard Johansen test
print("\n[2] Testing johansen()...")
try:
    result = johansen(X, model=2, k=2)
    print(f"    [OK] Johansen test passed")
    print(f"      Eigenvalues: {result.eigenvalues}")
    print(f"      Trace stats: {result.trace}")
    print(f"      Observations: {result.n_obs}")
except Exception as e:
    print(f"    [FAIL] Johansen test failed: {e}")

# Test 3: Johansen-Fourier test (single frequency)
print("\n[3] Testing johansen_fourier() - single frequency...")
try:
    result = johansen_fourier(X, model=3, k=2, f=1, option=1)
    print(f"    [OK] Johansen-Fourier (single) test passed")
    print(f"      Trace stats: {result.trace}")
    print(f"      CV (5%): {result.cv_trace}")
    print(f"      Cointegration rank: {result.get_cointegration_rank()}")
except Exception as e:
    print(f"    [FAIL] Johansen-Fourier test failed: {e}")

# Test 4: Johansen-Fourier test (cumulative frequency)
print("\n[4] Testing johansen_fourier() - cumulative frequency...")
try:
    result = johansen_fourier(X, model=3, k=2, f=2, option=2)
    print(f"    [OK] Johansen-Fourier (cumulative) test passed")
    print(f"      Trace stats: {result.trace}")
    print(f"      CV (5%): {result.cv_trace}")
except Exception as e:
    print(f"    [FAIL] Johansen-Fourier cumulative test failed: {e}")

# Test 5: SC-VECM test
print("\n[5] Testing sc_vecm()...")
try:
    result = sc_vecm(r=0, y=X, max_lag=4, lambda_L=0.1)
    print(f"    [OK] SC-VECM test passed")
    print(f"      Selected model: {result.selected_model}")
    print(f"      Trace (no break): {result.trace_no_break:.4f}")
    print(f"      Trace (break): {result.trace_break:.4f}")
    if result.break_location:
        print(f"      Break location: {result.break_location} (fraction: {result.break_fraction:.3f})")
except Exception as e:
    print(f"    [FAIL] SC-VECM test failed: {e}")

# Test 6: SBC test
print("\n[6] Testing sbc_test()...")
try:
    result = sbc_test(r=0, y=X, max_lag=4, lambda_L=0.1, f_max=3, option=2)
    print(f"    [OK] SBC test passed")
    print(f"      Selected model: {result.selected_model}")
    print(f"      Trace from selected: {result.trace:.4f}")
    print(f"      All SBC values: {result.all_sbc}")
except Exception as e:
    print(f"    [FAIL] SBC test failed: {e}")

# Test 7: Union test
print("\n[7] Testing union_test()...")
try:
    result = union_test(X, model=3, k=2, f=2, option=2, r=0)
    print(f"    [OK] Union test passed")
    print(f"      Reject H0: {result.reject_h0}")
    print(f"      Fourier trace: {result.fourier_trace:.4f}")
    print(f"      SC-VECM trace: {result.scvecm_trace:.4f}")
    print(f"      Scale factor: {result.scale_factor:.4f}")
except Exception as e:
    print(f"    [FAIL] Union test failed: {e}")

# Test 8: Output formatting
print("\n[8] Testing output formatting...")
try:
    result = johansen_fourier(X, model=3, k=2, f=1, option=1)
    latex = to_latex(result)
    md = to_markdown(result)
    print(f"    [OK] LaTeX export: {len(latex)} characters")
    print(f"    [OK] Markdown export: {len(md)} characters")
except Exception as e:
    print(f"    [FAIL] Output formatting failed: {e}")

# Test 9: Summary output
print("\n[9] Testing summary output...")
try:
    result = johansen_fourier(X, model=3, k=2, f=1, option=1)
    summary = result.summary()
    print(f"    [OK] Summary generation: {len(summary.splitlines())} lines")
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT:")
    print("=" * 60)
    print(result)
except Exception as e:
    print(f"    [FAIL] Summary output failed: {e}")

print("\n" + "=" * 60)
print("    ALL TESTS COMPLETED!")
print("=" * 60)
