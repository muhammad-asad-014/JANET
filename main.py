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
    
    # Derivative for Newton-Raphson
    f_prime_expr = diff(expr, main_var)
    f_prime_mp = lambdify(vars_found, f_prime_expr, modules=['mpmath'])
    
    return expr, vars_found, f_mp, f_np, f_prime_mp

# --- SOLVER ENGINE ---
def run_bisection(f, a, b, tol, max_iter=100):
    iterations = []
    a, b = mp.mpf(a), mp.mpf(b)
    
    if f(a) * f(b) >= 0:
        return None, "f(a) and f(b) must have opposite signs."

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        err = abs(b - a) / 2
        
        iterations.append({
            "Iter": i, "a": mp.nstr(a), "b": mp.nstr(b), 
            "Root Guess": mp.nstr(c), "f(c)": mp.nstr(fc), "Abs Error": mp.nstr(err)
        })
        
        if fc == 0 or err < mp.mpf(tol):
            break
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
            
    return iterations, None

def run_newton(f, df, x0, tol, max_iter=100):
    iterations = []
    x = mp.mpf(x0)
    
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        
        if dfx == 0:
            return None, "Derivative is zero. Newton's method failed."
        
        x_next = x - fx/dfx
        err = abs(x_next - x)
        
        iterations.append({
            "Iter": i, "x_n": mp.nstr(x), "f(x_n)": mp.nstr(fx), 
            "f'(x_n)": mp.nstr(dfx), "Abs Error": mp.nstr(err)
        })
        
        x = x_next
        if err < mp.mpf(tol):
            break
            
    return iterations, None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Settings")
    precision = st.slider("Decimal Precision", 15, 150, 50)
    mp.dps = precision
    max_iter = st.number_input("Max Iterations", value=50)

# --- MAIN UI ---
st.title("🔬 Numerical Analysis Solver")
user_input = st.text_input("Enter your equation f(x) = 0", "exp(x) * sin(x) - 1")

try:
    expr, vars_found, f_mp, f_np, f_prime_mp = get_math_objects(user_input)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Interpreted Equation")
        st.latex(f"f({vars_found[0]}) = {latex(expr)}")
        
        method = st.selectbox("Choose Numerical Method", ["Bisection", "Newton-Raphson"])
        
        if method == "Bisection":
            a_val = st.text_input("Interval Start (a)", value="0.0")
            b_val = st.text_input("Interval End (b)", value="1.0")
            tol = st.text_input("Tolerance", value="1e-15")
        else:
            x0_val = st.text_input("Initial Guess (x0)", value="0.5")
            tol = st.text_input("Tolerance", value="1e-15")

        submit = st.button("Calculate Iterations")

    if submit:
        with col2:
            st.subheader("Convergence Table")
            if method == "Bisection":
                iters, error = run_bisection(f_mp, a_val, b_val, tol, max_iter)
            else:
                iters, error = run_newton(f_mp, f_prime_mp, x0_val, tol, max_iter)
            
            if error:
                st.error(error)
            else:
                df_results = pd.DataFrame(iters)
                st.dataframe(df_results, use_container_width=True)
                
                final_root = iters[-1]["Root Guess"] if method == "Bisection" else iters[-1]["x_n"]
                st.success(f"Final Root: {final_root}")

                # Plotting
                fig, ax = plt.subplots()
                root_f = float(final_root)
                x_plot = np.linspace(root_f - 2, root_f + 2, 400)
                ax.plot(x_plot, f_np(x_plot), label="f(x)")
                ax.axhline(0, color='black', lw=1)
                ax.scatter([root_f], [0], color='red', zorder=5, label="Root")
                ax.legend()
                st.pyplot(fig)

except Exception as e:
    st.warning(f"Error: {e}")
