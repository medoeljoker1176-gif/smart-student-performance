"""
recommendations.py
------------------
Personalised, prioritised advice.

How it works (4 steps):
  1. RULES look at the student's own inputs and decide which habits are below a
     rule-of-thumb target (thresholds below are prototype choices, not standards).
  2. Each triggered rule gets a SEVERITY (0-1): how far below the target the student is.
  3. WHAT-IF: we ask the trained model "what would P(pass) be if this one habit reached
     the target?". The difference is the estimated GAIN. It shows association learned
     from synthetic data, not a guaranteed effect.
  4. Recommendations are ranked by gain (severity breaks ties) -> Priority 1, 2, 3...
"""
from dataclasses import dataclass
from typing import Callable, Optional

# feature: (critical_below, target). Below `critical` = big gap, below `target` = moderate gap.
NUMERIC_RULES = {
    "Study_Hours": (3, 4.5),
    "Attendance": (75, 85),
    "Previous_Score": (50, 65),
    "Assignment_Score": (50, 65),
}


@dataclass
class Recommendation:
    feature: str
    title: str
    action: str
    reason: str
    severity: float                    # 0..1
    gain: Optional[float] = None       # estimated change of P(pass) (0..1), None if not applicable
    what_if: Optional[str] = None      # readable "if X then Y"

    @property
    def rank_score(self) -> float:
        return (self.gain or 0.0) + 0.1 * self.severity


def _severity(value, target):
    return max(0.0, min(1.0, (target - value) / target))


def _candidates(s: dict) -> list:
    """Step 1 + 2: rule-based candidates (no model needed)."""
    recs = []
    study, att, prev, asg, sleep = (s["Study_Hours"], s["Attendance"], s["Previous_Score"],
                                    s["Assignment_Score"], s["Sleep_Hours"])

    if study < NUMERIC_RULES["Study_Hours"][1]:
        crit, target = NUMERIC_RULES["Study_Hours"]
        recs.append(Recommendation(
            "Study_Hours", "Increase focused study time",
            f"Build up to about {target:g} hours per day using short, focused sessions (e.g. 45 min study + 10 min break).",
            f"You study {study:g} h/day, below the {target:g} h target." + (" This is a large gap." if study < crit else ""),
            _severity(study, target)))
    elif study > 8:
        recs.append(Recommendation(
            "Study_Hours", "Protect your rest",
            "Very long study days can backfire. Cap study at about 6-8 hours and add breaks.",
            f"You study {study:g} h/day, which is unusually high.", 0.3))

    if att < NUMERIC_RULES["Attendance"][1]:
        crit, target = NUMERIC_RULES["Attendance"]
        recs.append(Recommendation(
            "Attendance", "Improve class attendance",
            f"Aim for at least {target:g}% attendance. Plan your week around lectures and tell your teacher early if you must miss one.",
            f"Your attendance is {att:g}%, below the {target:g}% target." + (" This is a large gap." if att < crit else ""),
            _severity(att, target)))

    if prev < NUMERIC_RULES["Previous_Score"][1]:
        crit, target = NUMERIC_RULES["Previous_Score"]
        recs.append(Recommendation(
            "Previous_Score", "Review weak topics and practise past exams",
            "List the topics you lost marks on last time, revise them first, then solve past papers under time limits.",
            f"Your previous score is {prev:g}/100, below the {target:g} target." + (" This is a large gap." if prev < crit else ""),
            _severity(prev, target)))

    if asg < NUMERIC_RULES["Assignment_Score"][1]:
        crit, target = NUMERIC_RULES["Assignment_Score"]
        recs.append(Recommendation(
            "Assignment_Score", "Improve assignment completion",
            "Start assignments earlier, submit on time, and ask your teacher for feedback on lost marks.",
            f"Your assignment score is {asg:g}/100, below the {target:g} target." + (" This is a large gap." if asg < crit else ""),
            _severity(asg, target)))

    if sleep < 7:
        recs.append(Recommendation(
            "Sleep_Hours", "Keep a regular sleep schedule",
            "Aim for 7-8 hours of sleep with fixed bed and wake times; sleep supports memory and focus.",
            f"You sleep {sleep:g} h/day, below the 7 h target." + (" This is a large gap." if sleep < 6 else ""),
            _severity(sleep, 7.5)))
    elif sleep > 9:
        recs.append(Recommendation(
            "Sleep_Hours", "Balance your sleep routine",
            "Oversleeping can cut into study time. Try a regular 7-8 hour routine.",
            f"You sleep {sleep:g} h/day, which is above the usual 7-9 h range.", 0.2))

    if s["Participation"] == "Low":
        recs.append(Recommendation(
            "Participation", "Participate more in class",
            "Ask or answer at least one question per class and join a study group.",
            "Your class participation is Low.", 0.5))

    if s["Internet_Access"] == "No":
        recs.append(Recommendation(
            "Internet_Access", "Secure access to online resources",
            "Use campus labs or the library for online learning material and assignment submission.",
            "You have no internet access at home.", 0.4))

    if s["Extracurricular"] == "Yes" and study < 3:
        recs.append(Recommendation(
            "Extracurricular", "Balance activities with study",
            "Keep your activities, but fix a weekly study timetable first so they do not replace study time.",
            "You do extracurricular activities and study less than 3 h/day.", 0.3))
    return recs


# What value to try in the what-if check for each feature
def _what_if_value(rec: Recommendation, s: dict):
    f = rec.feature
    if f in NUMERIC_RULES:
        target = NUMERIC_RULES[f][1]
        return target if s[f] < target else None
    if f == "Sleep_Hours":
        return 7.5 if (s[f] < 7 or s[f] > 9) else None
    if f == "Study_Hours" and s[f] > 8:
        return 6.0
    if f == "Participation":
        return "Medium"
    return None                      # internet / extracurricular: no what-if


def get_prioritized_recommendations(student: dict,
                                    p_pass_fn: Optional[Callable[[dict], float]] = None) -> list:
    """Ranked list of Recommendation objects (Priority 1 first).

    p_pass_fn: function(student_dict) -> P(pass). If given, each recommendation gets
    an estimated gain from the model's what-if answer.
    """
    recs = _candidates(student)
    if p_pass_fn is not None and recs:
        current = p_pass_fn(student)
        for rec in recs:
            new_value = _what_if_value(rec, student)
            if new_value is None:
                continue
            changed = {**student, rec.feature: new_value}
            new_p = p_pass_fn(changed)
            rec.gain = max(0.0, new_p - current)
            value_text = f"{new_value:g}" if isinstance(new_value, (int, float)) else str(new_value)
            rec.what_if = (f"If {rec.feature.replace('_', ' ').lower()} were {value_text}, the model's P(pass) "
                           f"would go from {current:.0%} to {new_p:.0%}.")
    recs.sort(key=lambda r: r.rank_score, reverse=True)
    return recs


def get_recommendations(s: dict) -> list:
    """Backward-compatible helper from the first version: plain list of advice strings."""
    recs = get_prioritized_recommendations(s)
    return [f"{r.title}: {r.action}" for r in recs] or ["Great habits! Keep up your routine and keep revising regularly."]
