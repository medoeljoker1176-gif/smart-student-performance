"""AI Explanation - why did the model predict this?"""
import streamlit as st

from src.prediction import format_probability
from src import charts, ui
from src.config import CATEGORY_VALUES, FEATURE_LABELS, FEATURES, INPUT_LIMITS
from src.prediction import p_pass_for
from src.validation import validate_student_input


def render():
    ui.hero("AI Explanation", "Understand why the model gave this probability - for one student and for the whole model.")
    bundle = ui.require_bundle()
    last = ui.require_last_analysis()
    expl, res = last["explanation"], last["result"]

    t1, t2, t3, t4 = st.tabs(["This prediction", "Global importance", "What-if simulator", "How to read this"])

    with t1:
        c = st.columns(3)
        c[0].metric("Average reference student", f"{expl['base']:.1%}", help="Mean P(pass) of the reference students")
        c[1].metric("This student", f"{expl['prediction']:.1%}", delta=f"{(expl['prediction'] - expl['base']) * 100:+.1f} points")
        c[2].metric("Risk level", f"{res.risk['icon']} {res.risk['level']}")
        st.plotly_chart(charts.contributions_bar(expl["table"]), key="expl_contrib")
        st.markdown("**In words:**")
        for sentence in expl["table"]["Sentence"]:
            st.write("- " + sentence)
        st.caption("The bars add up exactly to the difference between this student's probability and the average "
                   "reference student's probability.")

    with t2:
        metrics = ui.load_metrics()
        if not metrics:
            st.warning("Global importance is created during training. Run `python train_model.py`.")
        else:
            imp = metrics["global_importance"]
            st.plotly_chart(charts.global_importance_bar(imp), key="expl_global")
            top = list(imp)[:3]
            st.info("Across the tested students the three most influential factors were: "
                    + ", ".join(f"**{FEATURE_LABELS[f]}**" for f in top) + ".")
            st.caption("Computed on a sample of held-out students. It shows what THIS model uses, not what causes "
                       f"success in real life - the data is {ui.data_kind()}.")

    with t3:
        st.markdown("Change **one** factor of the analysed student and see how the model's probability changes.")
        feature = st.selectbox("Factor to change", FEATURES, format_func=lambda f: FEATURE_LABELS[f], key="whatif_feature")
        current = last["student"][feature]
        if feature in CATEGORY_VALUES:
            options = CATEGORY_VALUES[feature]
            new_value = st.selectbox("New value", options, index=options.index(current), key=f"whatif_{feature}_{res.p_pass:.4f}")
        else:
            lo, hi = INPUT_LIMITS[feature]
            if feature == "Age":
                new_value = st.number_input("New value", lo, hi, int(current), step=1, key=f"whatif_{feature}_{res.p_pass:.4f}")
            else:
                step = 1.0 if feature in ("Attendance", "Previous_Score", "Assignment_Score") else 0.5
                new_value = st.slider("New value", float(lo), float(hi), float(current), step, key=f"whatif_{feature}_{res.p_pass:.4f}")
        candidate = {**last["student"], feature: new_value}
        check = validate_student_input(candidate, bundle["training_ranges"])
        if ui.show_validation(check):
            new_p = p_pass_for(bundle, check.clean)
            c1, c2 = st.columns(2)
            c1.metric("Original P(pass)", f"{format_probability(res.p_pass)}")
            c2.metric("New P(pass)", f"{new_p:.1%}", delta=f"{(new_p - res.p_pass) * 100:+.1f} points")
            st.caption("This shows how the MODEL reacts. It is not proof that changing the habit causes the result.")

    with t4:
        st.markdown(
            "**What is being explained?** The model outputs a probability of passing. Each factor gets a "
            "*contribution*: how many percentage points it added to or removed from the probability, compared "
            "with an average reference student.\n\n"
            "**How is it calculated?** With **Shapley values** - the idea behind the SHAP method. For every possible "
            "combination of factors we ask the model for a prediction with those factors set to this student's values "
            "and the others replaced by reference students, then average each factor's marginal effect. "
            "With only 10 factors we compute it *exactly* (1,024 combinations) instead of approximating it.\n\n"
            "**Reading the chart:** green = pushes towards passing, red = pushes towards failing.\n\n"
            "**Correlation vs attribution vs causation:** *correlation* is how two columns move together in the data; "
            "*model attribution* (what this page shows) is how much the model relied on a factor for this prediction; "
            f"*causation* would mean changing the factor changes the real outcome, which this {ui.data_kind()} data and model "
            "cannot show.\n\n"
            "**Limits:** contributions describe the model's behaviour, not real-world causes.")
