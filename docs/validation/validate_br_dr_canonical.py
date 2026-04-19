"""Run BusinessReport / DiagnosticReport on canonical DiD datasets.

Writes ``docs/validation/br_dr_canonical_validation.md`` with the
full BR ``summary()`` + ``full_report()`` + selected ``to_dict()``
blocks for each dataset. The markdown output is the reviewable
artifact; compare it against canonical literature interpretations
and record any divergences in
``docs/validation/br_dr_canonical_findings.md``.

Purpose: BR/DR gap #4 (real-dataset validation) — synthetic-DGP
tests pass but we haven't checked whether the prose output matches
canonical interpretations of applied work.

Run via: ``python docs/validation/validate_br_dr_canonical.py``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from diff_diff import (
    BusinessReport,
    CallawaySantAnna,
    DifferenceInDifferences,
    SunAbraham,
)
from diff_diff.datasets import (
    load_card_krueger,
    load_castle_doctrine,
    load_mpdta,
)

OUT_PATH = Path(__file__).parent / "br_dr_canonical_validation.md"


def _section(title: str, level: int = 2) -> str:
    return "#" * level + " " + title + "\n"


def _fence(body: str, lang: str = "") -> str:
    return f"```{lang}\n{body.rstrip()}\n```\n"


def _dump_block(name: str, block: dict) -> str:
    return _fence(json.dumps(block, indent=2, default=str), "json")


def _card_krueger_section() -> str:
    """Card & Krueger (1994) minimum wage — classic 2x2 DiD.

    Canonical finding: no significant negative effect of NJ minimum-wage
    increase on fast-food employment; published ATT ~ +2.8 FTE or
    approximately 0.6 FTE per store depending on specification. CI
    includes zero; direction positive.
    """
    parts = [_section("Card & Krueger (1994): NJ/PA minimum wage", 2)]
    ck = load_card_krueger()
    # Reshape wide -> long per the docstring example.
    ck_long = ck.melt(
        id_vars=["store_id", "state", "treated"],
        value_vars=["emp_pre", "emp_post"],
        var_name="period",
        value_name="employment",
    )
    ck_long["post"] = (ck_long["period"] == "emp_post").astype(int)
    did = DifferenceInDifferences()
    fit = did.fit(ck_long, outcome="employment", treatment="treated", time="post")
    br = BusinessReport(
        fit,
        outcome_label="FTE employment",
        outcome_unit="FTE",
        outcome_direction="higher_is_better",
        business_question="Did the NJ minimum-wage increase reduce fast-food employment?",
        treatment_label="the NJ minimum-wage increase",
        auto_diagnostics=False,  # 2x2 PT needs manual column kwargs; run without for now
    )
    parts.append(
        "Data: NJ (treated, min wage $4.25 -> $5.05 on 1992-04-01) vs PA "
        "(control, $4.25 throughout). Outcome: full-time equivalent employment. "
        f"N={len(ck)} stores.\n\n"
    )
    parts.append(
        "Canonical interpretation: no significant disemployment effect of the "
        "minimum-wage increase; published ATT ~ +0.59 FTE (positive direction). "
        "The famous finding was that the CI included zero.\n\n"
    )
    parts.append(_section("BusinessReport.summary()", 3))
    parts.append(_fence(br.summary()))
    parts.append(_section("BusinessReport.full_report()", 3))
    parts.append(_fence(br.full_report(), "markdown"))
    parts.append(_section("BusinessReport.to_dict() - headline + assumption + caveats", 3))
    d = br.to_dict()
    parts.append(_dump_block("headline", d.get("headline", {})))
    parts.append(_dump_block("assumption", d.get("assumption", {})))
    parts.append(_dump_block("caveats", d.get("caveats", [])))
    parts.append("\n---\n")
    return "".join(parts)


def _mpdta_section() -> str:
    """Callaway-Sant'Anna benchmark (mpdta): county-level log employment
    under staggered minimum-wage increases.

    Canonical finding: CS aggregate ATT roughly -0.04 to -0.05 on log
    employment (i.e., ~4-5% employment decline for treated counties).
    Group-level ATT(g,t) shown in CS Figure 1.
    """
    parts = [_section("Callaway-Sant'Anna benchmark (mpdta)", 2)]
    df = load_mpdta()
    cs = CallawaySantAnna(base_period="universal")
    fit = cs.fit(
        df,
        outcome="lemp",
        unit="countyreal",
        time="year",
        first_treat="first_treat",
        aggregate="event_study",
    )
    br = BusinessReport(
        fit,
        outcome_label="Log employment",
        outcome_unit="log_points",
        outcome_direction="higher_is_better",
        business_question="Did minimum-wage increases reduce county employment?",
        treatment_label="the state-level minimum wage increase",
        data=df,
        outcome="lemp",
        unit="countyreal",
        time="year",
        first_treat="first_treat",
    )
    parts.append(
        "Data: simulated county-level panel from R `did` package (Callaway & "
        "Sant'Anna 2021), 2003-2007, staggered minimum-wage increases. Outcome: "
        "log employment (`lemp`).\n\n"
    )
    parts.append(
        "Canonical interpretation: CS aggregate ATT ~ -0.04 to -0.05 (log points) "
        "on treated counties; group-specific ATT(g,t) negative across cohorts. "
        "See CS (2021) Figures 1-2.\n\n"
    )
    parts.append(_section("BusinessReport.summary()", 3))
    parts.append(_fence(br.summary()))
    parts.append(_section("BusinessReport.full_report()", 3))
    parts.append(_fence(br.full_report(), "markdown"))
    parts.append(_section("BusinessReport.to_dict() - headline + assumption + caveats", 3))
    d = br.to_dict()
    parts.append(_dump_block("headline", d.get("headline", {})))
    parts.append(_dump_block("assumption", d.get("assumption", {})))
    parts.append(_dump_block("pre_trends", d.get("pre_trends", {})))
    parts.append(_dump_block("sensitivity", d.get("sensitivity", {})))
    parts.append(_dump_block("caveats", d.get("caveats", [])))
    parts.append("\n---\n")
    return "".join(parts)


def _castle_doctrine_section() -> str:
    """Cheng & Hoekstra (2013): staggered adoption of Castle Doctrine laws.

    Canonical finding: ~8% increase in homicide rates in adopting
    states; no deterrent effect on burglary or other crimes.
    """
    parts = [_section("Cheng & Hoekstra (2013): Castle Doctrine laws", 2)]
    df = load_castle_doctrine()
    # CS with never-treated as control; outcome = homicide rate.
    cs = CallawaySantAnna(base_period="universal", control_group="never_treated")
    fit = cs.fit(
        df,
        outcome="homicide_rate",
        unit="state",
        time="year",
        first_treat="first_treat",
        aggregate="event_study",
    )
    br = BusinessReport(
        fit,
        outcome_label="Homicide rate (per 100k)",
        outcome_unit="per 100k population",
        outcome_direction="lower_is_better",
        business_question=(
            "Did Castle Doctrine law adoption change state homicide rates?"
        ),
        treatment_label="Castle Doctrine law adoption",
        data=df,
        outcome="homicide_rate",
        unit="state",
        time="year",
        first_treat="first_treat",
    )
    parts.append(
        "Data: state-year panel, staggered Castle Doctrine law adoption 2005-2009. "
        "Outcome: homicide rate per 100k population.\n\n"
    )
    parts.append(
        "Canonical interpretation: Cheng & Hoekstra (2013) found ~8% increase in "
        "homicide rates in states that adopted Castle Doctrine (no deterrent "
        "effect; if anything, an escalation).\n\n"
    )
    parts.append(_section("BusinessReport.summary()", 3))
    parts.append(_fence(br.summary()))
    parts.append(_section("BusinessReport.full_report()", 3))
    parts.append(_fence(br.full_report(), "markdown"))
    parts.append(_section("BusinessReport.to_dict() - headline + assumption + caveats", 3))
    d = br.to_dict()
    parts.append(_dump_block("headline", d.get("headline", {})))
    parts.append(_dump_block("assumption", d.get("assumption", {})))
    parts.append(_dump_block("pre_trends", d.get("pre_trends", {})))
    parts.append(_dump_block("sensitivity", d.get("sensitivity", {})))
    parts.append(_dump_block("caveats", d.get("caveats", [])))
    parts.append("\n---\n")
    return "".join(parts)


def _castle_doctrine_sun_abraham_section() -> str:
    """Same Castle Doctrine dataset but run through Sun-Abraham, as a
    cross-estimator consistency check. If SA and CS narrate the same
    canonical finding differently, that's a BR/DR source-faithfulness
    issue.
    """
    parts = [_section("Castle Doctrine under Sun-Abraham (2021)", 2)]
    df = load_castle_doctrine()
    sa = SunAbraham()
    fit = sa.fit(
        df,
        outcome="homicide_rate",
        unit="state",
        time="year",
        first_treat="first_treat",
    )
    br = BusinessReport(
        fit,
        outcome_label="Homicide rate (per 100k)",
        outcome_unit="per 100k population",
        outcome_direction="lower_is_better",
        business_question=(
            "Did Castle Doctrine law adoption change state homicide rates?"
        ),
        treatment_label="Castle Doctrine law adoption",
        data=df,
        outcome="homicide_rate",
        unit="state",
        time="year",
        first_treat="first_treat",
    )
    parts.append(
        "Same dataset and research question; different estimator. Testing BR/DR "
        "cross-estimator narrative consistency.\n\n"
    )
    parts.append(_section("BusinessReport.summary()", 3))
    parts.append(_fence(br.summary()))
    parts.append(_section("BusinessReport.full_report()", 3))
    parts.append(_fence(br.full_report(), "markdown"))
    parts.append(_section("BusinessReport.to_dict() - headline + assumption", 3))
    d = br.to_dict()
    parts.append(_dump_block("headline", d.get("headline", {})))
    parts.append(_dump_block("assumption", d.get("assumption", {})))
    parts.append("\n---\n")
    return "".join(parts)


def main() -> int:
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    header = (
        "# BR / DR canonical-dataset validation\n\n"
        "Output of ``docs/validation/validate_br_dr_canonical.py``. Each section "
        "runs BusinessReport (and its auto-constructed DiagnosticReport) on a "
        "canonical DiD dataset and dumps summary + full_report + selected "
        "to_dict blocks. The purpose is to compare BR's prose output against "
        "published canonical interpretations and record divergences in "
        "``br_dr_canonical_findings.md``.\n\n"
        "This file is regenerable; do not hand-edit.\n\n"
        "Datasets covered: Card-Krueger (1994), mpdta (Callaway-Sant'Anna 2021 "
        "benchmark), Castle Doctrine (Cheng-Hoekstra 2013, both CS and SA).\n\n"
        "---\n\n"
    )

    sections = [header]
    for name, fn in (
        ("card_krueger", _card_krueger_section),
        ("mpdta", _mpdta_section),
        ("castle_doctrine_cs", _castle_doctrine_section),
        ("castle_doctrine_sa", _castle_doctrine_sun_abraham_section),
    ):
        print(f"Running {name} ...", file=sys.stderr)
        try:
            sections.append(fn())
        except Exception as exc:  # noqa: BLE001
            sections.append(
                _section(f"{name} (ERROR)", 2)
                + _fence(f"{type(exc).__name__}: {exc}")
                + "\n---\n"
            )
            print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)

    OUT_PATH.write_text("".join(sections))
    print(f"Wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
