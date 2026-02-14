import streamlit as st
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify, Symbol, E, pi, lambdify, diff, latex
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="High-Precision Numerical Lab", layout="wide")
mp.dps = 50  # Default high precision

# --- UTILITY FUNCTIONS ---
def clean_equation(user_str):
    user_str = user_str.replace('^', '**')
    # Implicit multiplication: 2x -> 2*x
    user_str = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', user_str)
    user_str = re.sub(r'(\))([a-zA-Z\(])', r'\1*\2', user_str)
    return user_str

def get_math_objects(user_eq):
    clean_str = clean_equation(user_eq)
    constants_dict = {'e': E, 'pi': pi}
    expr = sympify(clean_str, locals=constants_dict)
    vars_found = sorted(expr.atoms(Symbol), key=lambda s: s.name)
    # Numerical function (mpmath for logic, numpy for plotting)
    f_mp = lambdify(vars_found, expr, modules=['mpmath'])
    f_np = lambdify(vars_found, expr, modules=['numpy'])
    return expr, vars_found, f_mp, f_np

# --- SIDEBAR: GUIDELINES ---
with st.sidebar:
    st.header("📝 Writing Guidelines")
    st.markdown("""
    - **Power:** Use `x^2` or `x**2`
    - **Multiplication:** `2x` or `2*x`
    - **Constants:** Use `e` or `pi`
    - **Functions:** `sin(x)`, `exp(x)`, `log(x)`, `sqrt(x)`
    - **Example:** `e^x * sin(x) - 1`
    """)
    precision = st.slider("Decimal Precision", 15, 150, 50)
    mp.dps = precision

# --- MAIN UI ---
st.title("🔬 Numerical Analysis Solver")
user_input = st.text_input("Enter your equation f(x) = 0", "exp(x) * sin(x) - 1")

try:
    expr, vars_found, f_mp, f_np = get_math_objects(user_input)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Interpreted Equation")
        st.latex(f"f({', '.join([str(v) for v in vars_found])}) = {latex(expr)}")
        
        method = st.selectbox("Choose Numerical Method", ["Bisection", "Newton-Raphson"])
        
        # Dynamic inputs based on detected variables and method
        st.info(f"Detected variable: **{vars_found[0]}**")
        main_var = vars_found[0]
        
        if method == "Bisection":
            a = st.number_input("Interval Start (a)", value=0.0)
            b = st.number_input("Interval End (b)", value=1.0)
            tol = st.text_input("Tolerance", value="1e-15")
        else:
            x0 = st.number_input("Initial Guess (x0)", value=0.5)
            tol = st.text_input("Tolerance", value="1e-15")

        submit = st.button("Solve Equation")

    # --- CALCULATION LOGIC ---
    if submit:
        with col2:
            st.subheader("Results")
            try:
                if method == "Bisection":
                    root = mp.findroot(f_mp, (mp.mpf(str(a)), mp.mpf(str(b))), solver='bisect', tol=mp.mpf(tol))
                else:
                    # mpmath findroot defaults to Secant/Newton types
                    root = mp.findroot(f_mp, mp.mpf(str(x0)), tol=mp.mpf(tol))
                
                st.success(f"**Root found!**")
                st.code(mp.nstr(root, precision))
                
                # Plotting
                st.subheader("Function Visualization")
                x_vals = np.linspace(float(root)-2, float(root)+2, 400)
                y_vals = f_np(x_vals)
                
                fig, ax = plt.subplots()
                ax.plot(x_vals, y_vals, label="f(x)")
                ax.axhline(0, color='black', lw=1)
                ax.axvline(float(root), color='red', linestyle='--', label=f'Root: {float(root):.4f}')
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Solver Error: {e}. Try changing your interval or guess.")

except Exception as e:
    st.warning("Please enter a valid equation to begin.")
