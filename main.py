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
    main_var = vars_found[0]
    
    f_mp = lambdify(vars_found, expr, modules=['mpmath'])
    f_np = lambdify(vars_found, expr, modules=['numpy'])
    
    f_prime_expr = diff(expr, main_var)
    f_prime_mp = lambdify(vars_found, f_prime_expr, modules=['mpmath'])
    
    return expr, vars_found, f_mp, f_np, f_prime_mp

# --- SOLVER ENGINES ---

def run_bisection(f, a, b, tol, max_iter):
    iterations = []
    a, b = mp.mpf(a), mp.mpf(b)
    if f(a) * f(b) >= 0: return None, "f(a) and f(b) must have opposite signs."
    
    prev_c = None
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        
        # Calculate error as |current_c - prev_c|
        if prev_c is not None:
            err = abs(c - prev_c)
        else:
            err = "N/A" # First iteration has no previous c
            
        iterations.append({
            "Iter": i, "a": mp.nstr(a), "b": mp.nstr(b), 
            "Root Guess": mp.nstr(c), "f(c)": mp.nstr(fc), "Abs Error": mp.nstr(err) if err != "N/A" else err
        })
        
        # Convergence check
        if abs(fc) < mp.mpf(tol) or (prev_c is not None and err < mp.mpf(tol)):
            break
            
        if f(a) * fc < 0: b = c
        else: a = c
        prev_c = c
        
    return iterations, None

def run_regula_falsi(f, a, b, tol, max_iter):
    iterations = []
    a, b = mp.mpf(a), mp.mpf(b)
    if f(a) * f(b) >= 0: return None, "f(a) and f(b) must have opposite signs."

    prev_c = None
    for i in range(1, max_iter + 1):
        fa, fb = f(a), f(b)
        c = (a*fb - b*fa) / (fb - fa)
        fc = f(c)
        
        if prev_c is not None:
            err = abs(c - prev_c)
        else:
            err = "N/A"
            
        iterations.append({
            "Iter": i, "a": mp.nstr(a), "b": mp.nstr(b), 
            "Root Guess": mp.nstr(c), "f(c)": mp.nstr(fc),"f(a)": mp.nstr(fa),"f(b)": mp.nstr(fb), "Abs Error": mp.nstr(err) if err != "N/A" else err
        })
        
        if abs(fc) < mp.mpf(tol) or (prev_c is not None and err < mp.mpf(tol)):
            break
            
        if fa * fc < 0: b = c
        else: a = c
        prev_c = c
        
    return iterations, None

def run_secant(f, x0, x1, tol, max_iter):
    iterations = []
    x_prev, x_curr = mp.mpf(x0), mp.mpf(x1)
    for i in range(1, max_iter + 1):
        f_prev, f_curr = f(x_prev), f(x_curr)
        if f_curr - f_prev == 0: return None, "Division by zero in Secant method."
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        err = abs(x_next - x_curr)
        iterations.append({"Iter": i, "x_n": mp.nstr(x_curr), "f(x_n)": mp.nstr(f_curr), "Abs Error": mp.nstr(err)})
        if err < mp.mpf(tol): break
        x_prev, x_curr = x_curr, x_next
    return iterations, None

# --- FIXED-POINT ITERATION ---
# Note: User must input g(x) such that x = g(x)
def run_fixed_point(g, x0, tol, max_iter):
    iterations = []
    x = mp.mpf(x0)
    for i in range(1, max_iter + 1):
        x_next = g(x)
        err = abs(x_next - x)
        iterations.append({"Iter": i, "x_n": mp.nstr(x), "g(x_n)": mp.nstr(x_next), "Abs Error": mp.nstr(err)})
        if err < mp.mpf(tol): break
        x = x_next
    return iterations, None

def run_newton(f, df, x0, tol, max_iter):
    iterations = []
    x = mp.mpf(x0)
    
    for i in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if dfx == 0: return None, "Derivative is zero. Method failed."
        
        x_next = x - fx/dfx
        err = abs(x_next - x) # Difference between steps
        
        iterations.append({
            "Iter": i, "x_n": mp.nstr(x), "f(x_n)": mp.nstr(fx), 
            "Abs Error": mp.nstr(err)
        })
        
        x = x_next
        if err < mp.mpf(tol):
            break
            
    return iterations, None
    

# --- MAIN UI ---
st.title("🔬 JANET")
user_input = st.text_input("Equation f(x) = 0", "exp(x) * sin(x) - 1")

try:
    expr, vars_found, f_mp, f_np, f_prime_mp = get_math_objects(user_input)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        st.latex(f"f({vars_found[0]}) = {latex(expr)}")
        
        method = st.selectbox("Numerical Method", ["Bisection", "Regula Falsi", "Newton-Raphson", "Secant"])
        max_i = st.number_input("Number of Iterations", min_value=1, max_value=500, value=20)
        tol = st.text_input("Tolerance", value="1e-30")
        
        if method in ["Bisection", "Regula Falsi"]:
            a_val = st.text_input("Interval Start (a)", value="0.0")
            b_val = st.text_input("Interval End (b)", value="1.0")
        else:
            x0_val = st.text_input("Initial Guess (x0)", value="0.5")

        precision = st.slider("Display Precision", 15, 150, 50)
        mp.dps = precision
        submit = st.button("Run Algorithm")

    if submit:
        with col2:
            st.subheader("Convergence Table")
            if method == "Bisection":
                iters, error = run_bisection(f_mp, a_val, b_val, tol, max_i)
            elif method == "Regula Falsi":
                iters, error = run_regula_falsi(f_mp, a_val, b_val, tol, max_i)
            elif method== 'Secant':
                x0_s = st.text_input("First Guess x0", "0.0")
                x1_s = st.text_input("Second Guess x1", "1.0")
                iters, error = run_secant(f_mp, x0_s, x1_s, tol, max_i)
            # elif method== 'Fixed-Point':
            #      run_fixed_point(g, x0, tol, max_iter):
            #     st.warning("Note: Equation must be in form x = g(x)")
            #     iters, error = run_fixed_point(f_mp, x0_s, x1_s, tol, max_i)
            else:
                iters, error = run_newton(f_mp, f_prime_mp, x0_val, tol, max_i)
            
            if error:
                st.error(error)
            else:
                df_results = pd.DataFrame(iters)
                st.dataframe(df_results, use_container_width=True)
                
                final_root = iters[-1]["Root Guess"] if method in ["Bisection", "Regula Falsi"] else iters[-1]["x_n"]
                st.success(f"Final Root: {final_root}")

                # Plotting
                fig, ax = plt.subplots()
                root_f = float(mp.mpf(final_root))
                x_plot = np.linspace(root_f - 2, root_f + 2, 400)
                ax.plot(x_plot, f_np(x_plot), label="f(x)", color="#1f77b4")
                ax.axhline(0, color='black', lw=1)
                ax.scatter([root_f], [0], color='red', s=100, label=f"Root ≈ {root_f:.4f}")
                ax.set_title(f"Visualizing Root using {method}")
                ax.legend()
                st.pyplot(fig)

except Exception as e:
    st.warning(f"Enter a valid expression: {e}")
