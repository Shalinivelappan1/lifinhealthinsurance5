import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =====================================================
# Session State (CRITICAL)
# =====================================================
if "has_results" not in st.session_state:
    st.session_state.has_results = False

if "results_bundle" not in st.session_state:
    st.session_state.results_bundle = None

if "submission_content" not in st.session_state:
    st.session_state.submission_content = None

# =====================================================
# Page Config
# =====================================================
st.set_page_config(page_title="LiFin Health Insurance Lab", layout="wide")

st.title("🏥 LiFin Health Insurance Lab — Prof. Shalini Velappan, IIM Trichy")

st.markdown("""
## 🏠 The Household as a Risk-Bearing Economic Agent

This laboratory studies **health insurance not as a product**, but as a **risk-transfer architecture**.

The objective is **not** to minimize average medical expenditure.  
The objective is to **control the tail of the loss distribution** and **avoid financial ruin**.

This is a **computational experiment in decision-making under risk and solvency constraints**.

---

### This simulator demonstrates:

• Family floater vs separate parents policies  
• Room rent proportional deduction (Indian reality)  
• Deductibles, co-pay, sub-limits  
• Super top-up layers  
• No Claim Bonus (NCB)  
• Zonal pricing (city cost differences)  
• Distributional (tail) risk over 30 years  
• Probability of financial ruin relative to income & savings  

⚠️ This is a **teaching simulator**, not an actuarial pricing engine.
""")

# =====================================================
# Teaching Theory Blocks
# =====================================================
with st.expander("📘 Theory: Why Expected Value is the Wrong Objective"):
    st.markdown("""
In the presence of **bankruptcy / ruin constraints**, minimizing expected cost is **economically incorrect**.

Insurance changes:
• The **shape** of the loss distribution  
• Not just its **mean**

Rational agents must control **tail risk**, not averages.
""")

with st.expander("📘 Theory: Probability of Ruin"):
    st.markdown("""
We define:

> **Ruin = Lifetime medical cost > (Savings + X × Annual Income)**

This makes risk **relative to financial capacity**.

A rational household optimizes under a **solvency constraint**, not an expected value objective.
""")

with st.expander("📘 Theory: Layered Risk Transfer"):
    st.markdown("""
Deductible + Base Cover + Super Top-Up is a **layered risk-transfer structure**.

Such architectures dominate naive full-coverage designs in:
• **Tail risk reduction per rupee of premium**
""")

# =====================================================
# Sidebar: Financial Capacity
# =====================================================
st.sidebar.header("💰 Household Financial Capacity")

income = st.sidebar.number_input("Annual Household Income (₹)", value=1_200_000, step=100_000)
savings = st.sidebar.number_input("Current Savings (₹)", value=800_000, step=100_000)

ruin_multiple = st.sidebar.slider("Ruin if Medical Cost Exceeds (X × Income + Savings)", 1, 15, 5)
ruin_threshold = savings + ruin_multiple * income

st.sidebar.markdown(f"**Ruin Threshold = ₹ {ruin_threshold:,.0f}**")

# =====================================================
# Sidebar: Family Structure
# =====================================================
st.sidebar.header("👨‍👩‍👧‍👦 Family Structure")

include_spouse = st.sidebar.checkbox("Include Spouse", value=True)
num_children = st.sidebar.selectbox("Number of Children", [0, 1, 2], index=1)
num_parents = st.sidebar.selectbox("Number of Parents", [0, 1, 2], index=2)

members = ["Self"]
if include_spouse:
    members.append("Spouse")
for i in range(num_children):
    members.append(f"Child {i+1}")
for i in range(num_parents):
    members.append(f"Parent {i+1}")

st.sidebar.markdown(f"**Total Covered Members:** {len(members)}")

# =====================================================
# Sidebar: Stress Tests
# =====================================================
st.sidebar.header("🔥 Stress Tests")

stress_parent = st.sidebar.checkbox("Double Parent Risk")
stress_metro = st.sidebar.checkbox("Force Metro Zone")
stress_inflation = st.sidebar.checkbox("High Medical Inflation (12%)")

# =====================================================
# Sidebar: Zone
# =====================================================
st.sidebar.header("🌍 Treatment City Zone")

zone = st.sidebar.selectbox("City Zone", ["A (Metro)", "B (Tier-2)", "C (Small city)"])
zone_multiplier = {"A (Metro)": 1.4, "B (Tier-2)": 1.1, "C (Small city)": 0.9}[zone]

if stress_metro:
    zone_multiplier = 1.4

# =====================================================
# Sidebar: Medical Environment
# =====================================================
st.sidebar.header("📈 Medical Environment")

inflation = st.sidebar.slider("Medical Inflation (%)", 4.0, 15.0, 8.0) / 100
if stress_inflation:
    inflation = 0.12

years = 30

normal_min = st.sidebar.number_input("Normal Event Cost - Min (₹)", value=50_000, step=10_000)
normal_max = st.sidebar.number_input("Normal Event Cost - Max (₹)", value=300_000, step=50_000)

major_min = st.sidebar.number_input("Major Event Cost - Min (₹)", value=500_000, step=100_000)
major_max = st.sidebar.number_input("Major Event Cost - Max (₹)", value=2_500_000, step=100_000)

# =====================================================
# Sidebar: Risk Parameters
# =====================================================
st.sidebar.header("⚠️ Annual Risk Parameters")

p_normal_base = st.sidebar.slider("Prob. Normal Hosp (%) Non-Parents", 0.0, 20.0, 5.0) / 100
p_major_base = st.sidebar.slider("Prob. Major Illness (%) Non-Parents", 0.0, 10.0, 1.0) / 100

st.sidebar.markdown("**Parents Risk (User Controlled)**")
p_normal_parent = st.sidebar.slider("Parent: Prob. Normal Hosp (%)", 0.0, 30.0, 12.0) / 100
p_major_parent = st.sidebar.slider("Parent: Prob. Major Illness (%)", 0.0, 20.0, 4.0) / 100

if stress_parent:
    p_normal_parent *= 2
    p_major_parent *= 2

parent_cost_multiplier = st.sidebar.slider("Parent Cost Multiplier", 1.0, 3.0, 1.6)

# =====================================================
# Preset Policies
# =====================================================
PRESETS = {
    "Starter": {"cover": 500_000, "deductible": 100_000, "copay": 0.2, "room": 3000, "premium": 12_000, "topup": False},
    "Standard": {"cover": 1_000_000, "deductible": 50_000, "copay": 0.1, "room": 5000, "premium": 25_000, "topup": True},
    "Premium": {"cover": 2_500_000, "deductible": 0, "copay": 0.0, "room": 10000, "premium": 55_000, "topup": True},
}

# =====================================================
# Policy UI
# =====================================================
def policy_ui(label, default_preset):
    st.sidebar.header(f"📜 {label}")

    preset = st.sidebar.selectbox(
        f"{label} Preset",
        ["Custom", "Starter", "Standard", "Premium"],
        index=["Custom","Starter","Standard","Premium"].index(default_preset)
    )

    base = PRESETS[preset] if preset != "Custom" else None

    floater = st.sidebar.checkbox(f"{label}: Family Floater", value=True)

    cover = st.sidebar.number_input(f"{label}: Base Cover (₹)", value=(base["cover"] if base else 1_000_000), step=100_000)
    deductible = st.sidebar.number_input(f"{label}: Deductible (₹)", value=(base["deductible"] if base else 50_000), step=50_000)
    copay = st.sidebar.slider(f"{label}: Co-pay (%)", 0, 50, int((base["copay"]*100) if base else 10)) / 100
    room_limit = st.sidebar.number_input(f"{label}: Room Rent Limit (₹)", value=(base["room"] if base else 5000), step=1000)
    premium = st.sidebar.number_input(f"{label}: Annual Premium (₹)", value=(base["premium"] if base else 25_000), step=2000)

    st.sidebar.markdown(f"**{label}: Super Top-Up**")
    has_topup = st.sidebar.checkbox(f"{label}: Enable Super Top-Up", value=(base["topup"] if base else False))
    topup_cover = st.sidebar.number_input(f"{label}: Top-Up Cover (₹)", value=2_000_000, step=500_000)
    topup_threshold = st.sidebar.number_input(f"{label}: Top-Up Threshold (₹)", value=cover, step=100_000)
    topup_premium = st.sidebar.number_input(f"{label}: Top-Up Premium (₹)", value=10_000, step=1_000)

    st.sidebar.markdown(f"**{label}: No Claim Bonus**")
    ncb_rate = st.sidebar.slider(f"{label}: NCB increase per no-claim year (%)", 0, 50, 10) / 100
    ncb_cap = st.sidebar.slider(f"{label}: NCB Max Cap (% of base cover)", 0, 200, 100) / 100

    return {
        "floater": floater,
        "cover": cover,
        "deductible": deductible,
        "copay": copay,
        "room_limit": room_limit,
        "premium": premium,
        "has_topup": has_topup,
        "topup_cover": topup_cover,
        "topup_threshold": topup_threshold,
        "topup_premium": topup_premium,
        "ncb_rate": ncb_rate,
        "ncb_cap": ncb_cap
    }

policyA = policy_ui("Policy A", "Standard")
policyB = policy_ui("Policy B", "Premium")
# =====================================================
# Simulation Engine
# =====================================================
def simulate_year_for_member(is_parent):
    if is_parent:
        pN, pM, mult = p_normal_parent, p_major_parent, parent_cost_multiplier
    else:
        pN, pM, mult = p_normal_base, p_major_base, 1.0

    u = np.random.rand()
    if u < pM:
        cost = np.random.uniform(major_min, major_max) * mult
    elif u < pM + pN:
        cost = np.random.uniform(normal_min, normal_max) * mult
    else:
        cost = 0

    room = np.random.choice([3000, 5000, 8000, 12000]) if cost > 0 else 0
    return cost * zone_multiplier, room


def apply_policy(policy, total_claim, avg_room_cost, effective_cover):
    if total_claim <= 0:
        return 0

    # Room rent proportional deduction
    if avg_room_cost > policy["room_limit"]:
        ratio = policy["room_limit"] / avg_room_cost
    else:
        ratio = 1.0

    admissible = total_claim * ratio

    # Deductible
    remaining = max(admissible - policy["deductible"], 0)

    # Co-pay
    insurer_share = remaining * (1 - policy["copay"])

    # Base cover
    paid_by_base = min(insurer_share, effective_cover)
    leftover = insurer_share - paid_by_base

    # Super top-up
    paid_by_topup = 0
    if policy["has_topup"] and admissible > policy["topup_threshold"]:
        paid_by_topup = min(leftover, policy["topup_cover"])

    total_paid = paid_by_base + paid_by_topup
    oop = total_claim - total_paid

    return oop


def simulate_lifetime(policy):
    total_oop = 0
    ncb_multiplier = 1.0

    for t in range(years):
        infl = (1 + inflation) ** t
        yearly_claim = 0
        rooms = []

        for m in members:
            is_parent = "Parent" in m
            cost, room = simulate_year_for_member(is_parent)
            cost = cost * infl
            yearly_claim += cost
            if cost > 0:
                rooms.append(room)

        effective_cover = policy["cover"] * min(ncb_multiplier, 1 + policy["ncb_cap"])

        if yearly_claim > 0:
            avg_room = np.mean(rooms)
            oop = apply_policy(policy, yearly_claim, avg_room, effective_cover)
            total_oop += oop
            ncb_multiplier = 1.0
        else:
            ncb_multiplier *= (1 + policy["ncb_rate"])

    # Premiums
    total_oop += policy["premium"] * years
    if policy["has_topup"]:
        total_oop += policy["topup_premium"] * years

    return total_oop


# =====================================================
# Run Simulation Button
# =====================================================
st.markdown("## 🎲 Monte Carlo Simulation")

sims = st.slider("Number of Simulation Paths", 500, 4000, 2000, step=500)

if st.button("▶️ Run Simulation"):

    no_ins, polA, polB = [], [], []

    dummy_no_ins = {
        "cover": 0, "deductible": 0, "copay": 0,
        "room_limit": 1e12, "premium": 0,
        "has_topup": False, "topup_cover": 0,
        "topup_threshold": 0, "topup_premium": 0,
        "ncb_rate": 0, "ncb_cap": 0
    }

    for _ in range(sims):
        no_ins.append(simulate_lifetime(dummy_no_ins))
        polA.append(simulate_lifetime(policyA))
        polB.append(simulate_lifetime(policyB))

    st.session_state.results_bundle = {
        "no_ins": np.array(no_ins),
        "polA": np.array(polA),
        "polB": np.array(polB),
        "ruin_threshold": ruin_threshold
    }
    st.session_state.has_results = True


# =====================================================
# Display Results Persistently
# =====================================================
if st.session_state.has_results:

    no_ins = st.session_state.results_bundle["no_ins"]
    polA = st.session_state.results_bundle["polA"]
    polB = st.session_state.results_bundle["polB"]
    ruin_threshold = st.session_state.results_bundle["ruin_threshold"]

    def stats(x):
        return {
            "Mean": np.mean(x),
            "Median": np.median(x),
            "95%ile": np.percentile(x, 95),
            "99%ile": np.percentile(x, 99),
            "Worst": np.max(x)
        }

    s0 = stats(no_ins)
    sA = stats(polA)
    sB = stats(polB)

    # =====================================================
    # Distribution Plot
    # =====================================================
    st.markdown("## 📈 Distribution of Lifetime Medical Cost")

    fig, ax = plt.subplots()
    ax.hist(no_ins, bins=40, alpha=0.4, label="No Insurance")
    ax.hist(polA, bins=40, alpha=0.4, label="Policy A")
    ax.hist(polB, bins=40, alpha=0.4, label="Policy B")

    for data in [polA, polB]:
        ax.axvline(np.mean(data), linestyle="--")
        ax.axvline(np.percentile(data, 95), linestyle=":")
        ax.axvline(np.percentile(data, 99), linestyle="-.")

    ax.legend()
    ax.set_xlabel("30-year Total Cost (₹)")
    ax.set_ylabel("Frequency")
    ax.set_title("Lifetime Cost Distribution (Mean / 95% / 99% markers)")
    st.pyplot(fig)

    st.info("""
🧠 **Interpretation (Distributional View):**

Notice that the **mean changes only modestly**, but the **right tail (99% / worst case) collapses** under better insurance architectures.

This demonstrates:

> Insurance is not about optimizing averages.  
> It is about **destroying tail risk**.
""")

    # =====================================================
    # Probability of Ruin
    # =====================================================
    ruin_no = np.mean(no_ins > ruin_threshold)
    ruin_A = np.mean(polA > ruin_threshold)
    ruin_B = np.mean(polB > ruin_threshold)

    st.markdown("## 💥 Probability of Financial Ruin")

    probs = {
        "No Insurance": ruin_no,
        "Policy A": ruin_A,
        "Policy B": ruin_B
    }

    fig2, ax2 = plt.subplots()
    ax2.bar(probs.keys(), probs.values())
    ax2.set_ylabel("Probability")
    ax2.set_title("Probability that Lifetime Medical Cost > Ruin Threshold")
    st.pyplot(fig2)

    st.info(f"""
🧠 **Interpretation (Ruin View):**

Ruin Threshold = ₹ {ruin_threshold:,.0f}

A rational household should not ask:

> "Which policy minimizes expected cost?"

It should ask:

> "Which policy minimizes **probability of financial ruin**?"
""")
    # =====================================================
    # Decision Table (Formal, Academic)
    # =====================================================
    st.markdown("## 📋 Formal Decision Summary Table")

    decision_table = {
        "Metric": ["Mean", "Median", "95th Percentile", "99th Percentile", "Worst Case", "Probability of Ruin"],
        "No Insurance": [
            f"₹ {s0['Mean']:,.0f}",
            f"₹ {s0['Median']:,.0f}",
            f"₹ {s0['95%ile']:,.0f}",
            f"₹ {s0['99%ile']:,.0f}",
            f"₹ {s0['Worst']:,.0f}",
            f"{ruin_no:.1%}"
        ],
        "Policy A": [
            f"₹ {sA['Mean']:,.0f}",
            f"₹ {sA['Median']:,.0f}",
            f"₹ {sA['95%ile']:,.0f}",
            f"₹ {sA['99%ile']:,.0f}",
            f"₹ {sA['Worst']:,.0f}",
            f"{ruin_A:.1%}"
        ],
        "Policy B": [
            f"₹ {sB['Mean']:,.0f}",
            f"₹ {sB['Median']:,.0f}",
            f"₹ {sB['95%ile']:,.0f}",
            f"₹ {sB['99%ile']:,.0f}",
            f"₹ {sB['Worst']:,.0f}",
            f"{ruin_B:.1%}"
        ],
    }

    st.table(decision_table)

    st.info("""
🧠 **Proposition (Decision-Theoretic View):**

Under bankruptcy / ruin constraints, **expected value is not the correct objective**.

Rational choice must be based on:
• Tail risk (95%, 99%, worst case)
• Probability of crossing the **solvency boundary**
""")

    # =====================================================
    # Architecture Verdict
    # =====================================================
    st.markdown("## 🏛️ Architecture Dominance Verdict")

    if ruin_A < ruin_B:
        better = "Policy A"
    else:
        better = "Policy B"

    st.success(f"""
### ✅ Dominant Architecture (in Ruin-Risk Space): **{better}**

This policy **dominates** because it delivers:

• Lower tail risk  
• Lower probability of ruin  
• Better robustness per rupee of premium  

This is a **robustness criterion**, not a return optimization criterion.
""")

    st.markdown("""
🧠 **Interpretation:**

Insurance is not an investment.  
It is a **convexity transfer device** that removes catastrophic states from your feasible set.
""")

    # =====================================================
    # How to Use This Lab (Teaching Script)
    # =====================================================
    st.markdown("## 🧪 Suggested Experiments")

    st.markdown("""
Try the following controlled experiments:

1) Set **Policy A** = Large family floater  
2) Set **Policy B** = Smaller floater + strong top-up  
3) Increase **parent risk** and **medical inflation**

Observe:
• Mean barely changes  
• Tail and ruin probability change **dramatically**

This demonstrates why **architecture beats coverage size**.
""")

    # =====================================================
    # Assignment & Submission
    # =====================================================
    st.markdown("## 📝 Student Research Note Submission")

    st.markdown("""
You are writing a **mini research note** titled:

> *Distributional and Ruin-Based Evaluation of Health Insurance Architectures*

Answer in complete sentences.
""")

    name = st.text_input("Name")
    roll = st.text_input("Roll No")

    q1 = st.text_area("1) Which architecture dominates in tail-risk space, and why?")
    q2 = st.text_area("2) Why is expected value an inappropriate objective in this problem?")
    q3 = st.text_area("3) Explain the economic role of room-rent proportional deduction.")
    q4 = st.text_area("4) Why does layered insurance (base + top-up) dominate naive coverage?")
    q5 = st.text_area("5) Explain 'probability of ruin' as a decision criterion.")
    q6 = st.text_area("6) What happens to the decision if parent risk or inflation increases?")

    if st.button("📄 Generate Submission File"):

        st.session_state.submission_content = f"""
LiFin Health Insurance Lab
Distributional and Ruin-Based Evaluation of Health Insurance Architectures
Date: {datetime.now()}

Name: {name}
Roll No: {roll}

===========================
Model & Financial Capacity
===========================
Annual Income: ₹ {income:,.0f}
Savings: ₹ {savings:,.0f}
Ruin Threshold: ₹ {ruin_threshold:,.0f}

===========================
Decision Summary Table
===========================
{decision_table}

===========================
Probability of Ruin
===========================
No Insurance: {ruin_no:.2%}
Policy A: {ruin_A:.2%}
Policy B: {ruin_B:.2%}

===========================
Answers
===========================
Q1: {q1}

Q2: {q2}

Q3: {q3}

Q4: {q4}

Q5: {q5}

Q6: {q6}
"""

        st.success("Submission file generated. Use the download button below.")

    if st.session_state.submission_content is not None:
        st.download_button(
            label="⬇️ Download Submission",
            data=st.session_state.submission_content,
            file_name=f"{name}_health_insurance_assignment.txt",
            mime="text/plain"
        )

# =====================================================
# Footer
# =====================================================
st.markdown("""
---
⚠️ This is a **teaching simulator**, not a pricing or underwriting engine.

It is designed to illustrate **risk, robustness, and ruin under uncertainty**.
""")
