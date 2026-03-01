import streamlit as st
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import sympify, Symbol, E, pi, lambdify, diff, latex
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="JANET", layout="wide")

# --- UTILITY FUNCTIONS ---
def clean_equation(user_str):
    user_str = user_str.replace('^', '**')
    user_str = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', user_str)
    user_str = re.sub(r'(\))([a-zA-Z\(])', r'\1*\2', user_str)
    return user_str

def get_math_objects(user_eq):
    clean_str = clean_equation(user_eq)
    constants_dict = {'e': E, 'pi': pi}
    expr = sympify(clean_str, locals=constants_dict)
    vars_found = sorted(expr.atoms(Symbol), key=lambda s: s.name)
    main_var = vars_found[0] if vars_found else Symbol('x')
    
    f_mp = lambdify(vars_found, expr, modules=['mpmath'])
    f_np = lambdify(vars_found, expr, modules=['numpy'])
    
    f_prime_expr = diff(expr, main_var)
    f_prime_mp = lambdify(vars_found, f_prime_expr, modules=['mpmath'])
    
    return expr, vars_found, f_mp, f_np, f_prime_mp

# --- SOLVER ENGINES ---

def run_bisection(f, a, b, tol, max_iter):
    iterations, a, b, prev_c = [], mp.mpf(a), mp.mpf(b), None
    fa, fb = f(a), f(b)
    if fa * fb >= 0: return None, "f(a) and f(b) must have opposite signs."
    
    for i in range(1, max_iter + 1):
        fa, fb = f(a), f(b)
        c = (a + b) / 2
        fc = f(c)
        err = abs(c - prev_c) if prev_c is not None else "N/A"
        
        iterations.append({
            "Iter": i, "a": mp.nstr(a), "b": mp.nstr(b), 
            "f(a)": mp.nstr(fa), "f(b)": mp.nstr(fb),
            "Guess (c)": mp.nstr(c), "f(c)": mp.nstr(fc), 
            "Abs Error": mp.nstr(err) if err != "N/A" else "N/A"
        })
        
        if abs(fc) < mp.mpf(tol) or (prev_c is not None and err < mp.mpf(tol)): break
        
        if fa * fc < 0: b = c
        else: a = c
        prev_c = c
    return iterations, None

def run_regula_falsi(f, a, b, tol, max_iter):
    iterations, a, b, prev_c = [], mp.mpf(a), mp.mpf(b), None
    if f(a) * f(b) >= 0: return None, "f(a) and f(b) must have opposite signs."
    
    for i in range(1, max_iter + 1):
        fa, fb = f(a), f(b)
        c = (a*fb - b*fa) / (fb - fa)
        fc = f(c)
        err = abs(c - prev_c) if prev_c is not None else "N/A"
        
        iterations.append({
            "Iter": i, "a": mp.nstr(a), "b": mp.nstr(b), 
            "f(a)": mp.nstr(fa), "f(b)": mp.nstr(fb),
            "Guess (c)": mp.nstr(c), "f(c)": mp.nstr(fc), 
            "Abs Error": mp.nstr(err) if err != "N/A" else "N/A"
        })
        
        if abs(fc) < mp.mpf(tol) or (prev_c is not None and err < mp.mpf(tol)): break
        
        if fa * fc < 0: b = c
        else: a = c
        prev_c = c
    return iterations, None

def run_secant(f, x0, x1, tol, max_iter):
    iterations, x_prev, x_curr = [], mp.mpf(x0), mp.mpf(x1)
    for i in range(1, max_iter + 1):
        f_prev, f_curr = f(x_prev), f(x_curr)
        if f_curr - f_prev == 0: return None, "Division by zero in Secant method."
        
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        err = abs(x_next - x_curr)
        
        iterations.append({
            "Iter": i, 
            "x_{n-1}": mp.nstr(x_prev), "x_n": mp.nstr(x_curr), 
            "f(x_{n-1})": mp.nstr(f_prev), "f(x_n)": mp.nstr(f_curr), 
            "Abs Error": mp.nstr(err)
        })
        
        if err < mp.mpf(tol): break
        x_prev, x_curr = x_curr, x_next
    return iterations, None

def run_newton(f, df, x0, tol, max_iter):
    iterations, x = [], mp.mpf(x0)
    for i in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if dfx == 0: return None, "Derivative is zero. Method failed."
        x_next = x - fx/dfx
        err = abs(x_next - x)
        iterations.append({"Iter": i, "x_n": mp.nstr(x), "f(x_n)": mp.nstr(fx), "Abs Error": mp.nstr(err)})
        if err < mp.mpf(tol): break
        x = x_next
    return iterations, None

def run_fixed_point(g, dg, x0, tol, max_iter):
    iterations, x_curr = [], mp.mpf(x0)
    slope = abs(dg(x_curr))
    if slope >= 1: st.warning(f"⚠️ Divergence Warning: |g'(x0)| = {mp.nstr(slope, 4)}.")
    else: st.info(f"✅ Stability Check: |g'(x0)| = {mp.nstr(slope, 4)}.")
    
    for i in range(1, max_iter + 1):
        x_next = g(x_curr)
        err = abs(x_next - x_curr)
        iterations.append({"Iter": i, "x_n": mp.nstr(x_curr), "g(x_n)": mp.nstr(x_next), "Abs Error": mp.nstr(err) if i > 1 else "N/A"})
        if i > 1 and err < mp.mpf(tol): break
        x_curr = x_next
    return iterations, None

# --- MAIN UI ---
st.sidebar.header("📝 Writing Guidelines")
st.sidebar.markdown("- **Power:** `x^2`\n- **Implicit:** `2x`\n- **Constants:** `e`, `pi`")
precision = st.sidebar.slider("Precision", 15, 150, 50)
mp.dps = precision

st.title("🔬 JANET: Root-Finding Lab")
user_input = st.text_input("Enter Equation f(x) or g(x)", "e^x * sin(x) - 1")

try:
    expr, vars_found, f_mp, f_np, f_prime_mp = get_math_objects(user_input)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        st.latex(f"f({vars_found[0]}) = {latex(expr)}")
        method = st.selectbox("Method", ["Bisection", "Regula Falsi", "Newton-Raphson", "Secant", "Fixed-Point"])
        
        if method in ["Bisection", "Regula Falsi", "Secant"]:
            p1 = st.text_input("Start (a) / x_{n-1}", "0.0")
            p2 = st.text_input("End (b) / x_n", "1.0")
        else:
            p1 = st.text_input("Initial Guess (x0)", "0.5")
            
        max_i = st.number_input("Iterations", 1, 500, 25)
        tol = st.text_input("Tolerance", "1e-30")
        submit = st.button("Calculate")

    if submit:
        with col2:
            st.subheader("Detailed Iterations")
            if method == "Bisection": iters, error = run_bisection(f_mp, p1, p2, tol, max_i)
            elif method == "Regula Falsi": iters, error = run_regula_falsi(f_mp, p1, p2, tol, max_i)
            elif method == "Newton-Raphson": iters, error = run_newton(f_mp, f_prime_mp, p1, tol, max_i)
            elif method == "Secant": iters, error = run_secant(f_mp, p1, p2, tol, max_i)
            elif method == "Fixed-Point": iters, error = run_fixed_point(f_mp, f_prime_mp, p1, tol, max_i)

            if error: st.error(error)
            elif iters:
                st.dataframe(pd.DataFrame(iters), use_container_width=True)
                
                # Dynamic Final Result extraction
                res_key = "Guess (c)" if method in ["Bisection", "Regula Falsi"] else ("g(x_n)" if method == "Fixed-Point" else "x_n")
                final_root = iters[-1][res_key]
                st.success(f"**Root Found:** {final_root}")

                fig, ax = plt.subplots()
                rf = float(mp.mpf(final_root))
                xp = np.linspace(rf - 2, rf + 2, 400)
                ax.plot(xp, f_np(xp), label="Function Curve")
                ax.axhline(0, color='black', lw=1)
                ax.scatter([rf], [0], color='red', label=f"Root: {rf:.4f}")
                ax.legend()
                st.pyplot(fig)

except Exception as e:
    st.info("Please enter a valid mathematical expression.")
