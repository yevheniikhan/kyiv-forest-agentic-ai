"""A minimal, transparent agentic workflow for biomass assessment.

Components
----------
* Tool registry: typed Python functions with a short description. The planner may call only these.
* Planner: proposes an ordered list of tool calls. By default a scripted planner is used, so the
  notebook runs without any API key. An LLM planner can be plugged in (see `llm_planner`).
* Deterministic checks: rules written by the researcher; they run after every step and do not
  depend on the language model.
* Approval gates: the researcher confirms the plan, the data and the final result before export.
* Run log: every step, argument, check and decision is written to a JSON file.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]


@dataclass
class Step:
    tool: str
    args: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


class Registry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, name, description):
        def deco(fn):
            self.tools[name] = Tool(name, description, fn)
            return fn
        return deco

    def catalogue(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())


class RunLog:
    def __init__(self, path="run_log.json"):
        self.path, self.events = path, []

    def add(self, kind, **payload):
        self.events.append({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **payload})

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2, default=str)
        return self.path


def approval_gate(question: str, summary: str, log: RunLog, auto: bool | None = None) -> bool:
    """Ask the researcher for a decision. `auto=True` is used only for unattended test runs."""
    print("\n" + "=" * 70 + f"\nРІШЕННЯ ДОСЛІДНИКА: {question}\n" + summary + "\n" + "=" * 70)
    if auto is None:
        answer = input("Підтвердити? [так/ні]: ").strip().lower()
        ok = answer in {"так", "т", "y", "yes"}
    else:
        ok = bool(auto)
        print("(автоматичне підтвердження для тестового запуску)" if ok else "(відхилено)")
    log.add("approval", question=question, approved=ok)
    return ok


def scripted_planner(goal: str, registry: Registry) -> list[Step]:
    """Fixed plan used for the demonstration. It mirrors what an LLM planner is expected to propose."""
    return [
        Step("load_samples", {}, "відбитки GEDI L4A і предиктори в межах лісу"),
        Step("train_validate", {"scheme": "random"}, "випадкова крос-валідація (оптимістична оцінка)"),
        Step("train_validate", {"scheme": "spatial"}, "просторова блокова крос-валідація"),
        Step("map_and_totals", {}, "карта біомаси, площа, запас вуглецю"),
        Step("compare_region", {}, "порівняння з регіональними значеннями 2011 року"),
    ]


def llm_planner(goal: str, registry: Registry, client=None, model="claude-sonnet-5-5") -> list[Step]:
    """Ask an LLM for a plan in JSON. The answer is parsed and every tool name is checked against
    the registry; unknown tools are rejected before anything runs."""
    prompt = (
        "Ти плануєш аналіз для дослідника-лісівника. Доступні інструменти:\n"
        f"{registry.catalogue()}\n\nМета: {goal}\n"
        'Поверни лише JSON-список кроків виду [{"tool": "...", "args": {...}, "note": "..."}].'
    )
    msg = client.messages.create(model=model, max_tokens=800, messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text
    steps = [Step(**s) for s in json.loads(text[text.index("["): text.rindex("]") + 1])]
    unknown = [s.tool for s in steps if s.tool not in registry.tools]
    if unknown:
        raise ValueError(f"План містить невідомі інструменти: {unknown}")
    return steps


def run(goal: str, registry: Registry, checks: dict[str, Callable[[dict], list[CheckResult]]],
        planner=scripted_planner, auto: bool | None = None, log_path="run_log.json"):
    log = RunLog(log_path)
    plan = planner(goal, registry)
    log.add("plan", goal=goal, steps=[asdict(s) for s in plan])
    summary = "\n".join(f"{i+1}. {s.tool} {s.args or ''} – {s.note}" for i, s in enumerate(plan))
    if not approval_gate("план аналізу", summary, log, auto):
        log.save()
        return None

    state: dict[str, Any] = {}
    for i, step in enumerate(plan, 1):
        tool = registry.tools[step.tool]
        t0 = time.time()
        out = tool.fn(state, **step.args)
        state[f"{step.tool}:{step.args.get('scheme', '')}".rstrip(":")] = out
        log.add("step", n=i, tool=step.tool, args=step.args, seconds=round(time.time() - t0, 2),
                output={k: v for k, v in (out or {}).items() if isinstance(v, (int, float, str))})
        results = checks.get(step.tool, lambda s, o: [])(state, out)
        for r in results:
            log.add("check", step=i, **asdict(r))
            print(("  ✓ " if r.passed else "  ✗ ") + f"{r.name}: {r.detail}")
        if any(not r.passed for r in results):
            failed = "; ".join(r.name for r in results if not r.passed)
            if not approval_gate(f"крок {i} не пройшов перевірки ({failed}). Продовжити?",
                                 "Агент зупинився. Рішення про продовження ухвалює дослідник.", log, auto):
                log.save()
                return state
        if step.tool == "load_samples":
            if not approval_gate("дані для навчання", f"n = {out.get('n')}", log, auto):
                log.save()
                return state

    approval_gate("результат перед експортом", json.dumps(
        {k: v for k, v in state.get("map_and_totals", {}).items() if isinstance(v, (int, float))},
        ensure_ascii=False, indent=1), log, auto)
    print("Журнал запуску:", log.save())
    return state
