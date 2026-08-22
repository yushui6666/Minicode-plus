from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from minicode.skills import SkillSummary, discover_skills, extract_description

HOTLIST_FILENAME = "_hotlist.md"
USAGE_FILENAME = "_usage.json"
DEFAULT_TOP_N = 20
DECAY_PER_DAY = 0.2
CURATE_EVERY_N_USES = 5
CURATE_MIN_INTERVAL_SEC = 60 * 30  # 30 min
COLD_START_DAYS = 7


@dataclass(slots=True)
class SkillUsage:
    name: str
    count: int = 0
    last_used: float = 0.0  # epoch seconds
    first_seen: float = 0.0
    description: str = ""


def _usage_path(cwd: str | Path) -> Path:
    return Path(cwd) / ".mini-code" / "skills" / USAGE_FILENAME


def _hotlist_path(cwd: str | Path) -> Path:
    return Path(cwd) / ".mini-code" / "skills" / HOTLIST_FILENAME


def _now() -> float:
    return time.time()


def _decay_score(usage: SkillUsage, now: float | None = None) -> float:
    if now is None:
        now = _now()
    days_ago = max(0.0, (now - usage.last_used) / 86400.0) if usage.last_used else 0.0
    return usage.count / (1.0 + days_ago * DECAY_PER_DAY)


def _load_usage_raw(cwd: str | Path) -> dict[str, Any]:
    p = _usage_path(cwd)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_usage(cwd: str | Path) -> dict[str, SkillUsage]:
    raw = _load_usage_raw(cwd)
    out: dict[str, SkillUsage] = {}
    for name, v in raw.items():
        if not isinstance(v, dict):
            continue
        try:
            out[name] = SkillUsage(
                name=name,
                count=int(v.get("count", 0) or 0),
                last_used=float(v.get("last_used", 0) or 0),
                first_seen=float(v.get("first_seen", 0) or 0),
                description=str(v.get("description", "") or ""),
            )
        except Exception:
            continue
    return out


def _save_usage(cwd: str | Path, usage: dict[str, SkillUsage]) -> None:
    p = _usage_path(cwd)
    p.parent.mkdir(parents=True, exist_ok=True)
    serial = {k: asdict(v) for k, v in usage.items()}
    # atomic write via temp
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def record_skill_use(cwd: str | Path, name: str, description: str = "") -> None:
    """Record one use of a skill. Fail-open: never breaks load_skill."""
    try:
        norm = name.strip()
        if not norm:
            return
        usage = load_usage(cwd)
        now = _now()
        u = usage.get(norm)
        if u is None:
            u = SkillUsage(name=norm, count=0, last_used=now, first_seen=now, description=description)
        u.count += 1
        u.last_used = now
        if description:
            u.description = description
        if not u.first_seen:
            u.first_seen = now
        usage[norm] = u
        _save_usage(cwd, usage)
        _maybe_curate(cwd, usage)
    except Exception:
        return


def _maybe_curate(cwd: str | Path, usage: dict[str, SkillUsage] | None = None) -> None:
    try:
        if usage is None:
            usage = load_usage(cwd)
        total_uses = sum(u.count for u in usage.values())
        # curate every N uses
        if total_uses % CURATE_EVERY_N_USES != 0 and total_uses != 1:
            # also check time interval as fallback
            hp = _hotlist_path(cwd)
            if hp.exists():
                age = _now() - hp.stat().st_mtime
                if age < CURATE_MIN_INTERVAL_SEC:
                    return
            else:
                # if no hotlist yet, curate immediately on first few uses
                if total_uses > 1:
                    return
        curate_hotlist(cwd)
    except Exception:
        return


def curate_hotlist(cwd: str | Path, top_n: int = DEFAULT_TOP_N) -> Path:
    """Generate _hotlist.md with TopN by decay_score, with cold-start protection."""
    cwd_path = Path(cwd)
    usage = load_usage(cwd_path)
    discovered = discover_skills(cwd_path)
    now = _now()

    # Build map name->description from discovered (most authoritative)
    discovered_desc = {s.name: s.description for s in discovered}
    discovered_names = {s.name for s in discovered}

    # Ensure usage entries for discovered skills exist (cold start)
    for s in discovered:
        if s.name not in usage:
            # not used yet, but give it a cold-start score
            usage[s.name] = SkillUsage(
                name=s.name, count=0, last_used=0, first_seen=now, description=s.description
            )
        else:
            # keep description fresh
            if discovered_desc.get(s.name):
                usage[s.name].description = discovered_desc[s.name]

    # Compute decay scores
    scored: list[tuple[float, SkillUsage]] = []
    for name, u in usage.items():
        if name not in discovered_names:
            # stale entry for removed skill, keep but deprioritize
            continue
        score = _decay_score(u, now)
        # cold-start boost: if first_seen within COLD_START_DAYS and count==0, give small epsilon
        if u.count == 0 and (now - u.first_seen) / 86400.0 < COLD_START_DAYS:
            score = max(score, 0.01)
        scored.append((score, u))

    scored.sort(key=lambda x: (-x[0], x[1].name))

    top = scored[:top_n]
    # If more than top_n discovered but scored less, ensure at least 3 cold-start not starved:
    # already covered by epsilon, but if hotlist is full of high-score, cold ones naturally at bottom

    hotlist_path = _hotlist_path(cwd_path)
    hotlist_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Hot Skills (auto-curated {time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}, Top{top_n} by decay_score)")
    lines.append("")
    lines.append(f"Total discovered: {len(discovered)} | Total uses: {sum(u.count for u in usage.values())} | Decay: 1/(1+days*0.2)")
    lines.append("")
    lines.append("| rank | name | description | uses | decay_score | last_used |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for idx, (score, u) in enumerate(top, start=1):
        last = time.strftime("%Y-%m-%d", time.localtime(u.last_used)) if u.last_used else "-"
        desc = (u.description or discovered_desc.get(u.name, "")).replace("|", "/").strip()
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(f"| {idx} | {u.name} | {desc} | {u.count} | {score:.2f} | {last} |")
    lines.append("")
    lines.append("Tip: Model should prefer these 20, but may still call load_skill for any discovered skill via /skills.")
    lines.append("")

    tmp = hotlist_path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(hotlist_path)
    return hotlist_path


def _tokenize(text: str) -> list[str]:
    """Split text into ASCII words plus individual CJK characters."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    return tokens + cjk


def rerank_hot_skills(skills: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    """Re-rank hot skills against the current task query with a small BM25.

    Fail-open: any problem returns the original order unchanged.
    A skill whose name appears verbatim in the query gets a strong boost,
    so an explicit \"use the code-review skill\" request always wins.
    """
    if not query or not skills:
        return skills
    try:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return skills
        docs = [
            _tokenize(f"{s.get('name', '')} {s.get('description', '')}")
            for s in skills
        ]
        n_docs = len(docs)
        avgdl = sum(len(d) for d in docs) / max(1, n_docs)
        k1 = 1.5
        b = 0.75
        df: dict[str, int] = {}
        for doc in docs:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1

        def _score(doc: list[str], name: str) -> float:
            dl = len(doc)
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            total = 0.0
            for token in q_tokens:
                tf = counts.get(token)
                if not tf:
                    continue
                idf = math.log(1 + (n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
                total += idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / max(avgdl, 1.0))
                )
            lowered_query = query.lower()
            if name and name.lower() in lowered_query:
                total += 5.0
            return total

        scored = [(_score(doc, skills[i].get("name", "")), i) for i, doc in enumerate(docs)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [skills[i] for _, i in scored]
    except Exception:
        return skills


def get_hot_skills_for_prompt(cwd: str | Path, query: str | None = None) -> list[dict[str, str]]:
    """Return Top20 hot skills as dicts for prompt injection. Fallback to discover_skills."""
    cwd_path = Path(cwd)
    hp = _hotlist_path(cwd_path)
    if not hp.exists():
        # try curate lazily if usage exists or discovered exists
        try:
            curate_hotlist(cwd_path)
        except Exception:
            pass
    if hp.exists():
        try:
            text = hp.read_text(encoding="utf-8")
            # parse markdown table
            out: list[dict[str, str]] = []
            for line in text.splitlines():
                line=line.strip()
                if not line.startswith("|") or line.startswith("| rank"):
                    continue
                if line.startswith("| ---"):
                    continue
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) < 6:
                    continue
                # parts: rank, name, description, uses, decay_score, last_used
                name = parts[1]
                desc = parts[2]
                if not name or name == "name":
                    continue
                out.append({"name": name, "description": desc})
            if out:
                return rerank_hot_skills(out, query or "")
        except Exception:
            pass
    # fallback: full discover but capped to 20 for prompt brevity
    discovered = discover_skills(cwd_path)
    out = [{"name": s.name, "description": s.description} for s in discovered]
    out = rerank_hot_skills(out, query or "")
    return out[:DEFAULT_TOP_N]


def get_hotlist_text(cwd: str | Path) -> str | None:
    p = _hotlist_path(cwd)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None