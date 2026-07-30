import asyncio
import uuid
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.fetchers import FetchError, fetch_github_org_repos, fetch_page_text
from app.llm import call_claude_json, call_claude_text, parse_markdown_bullets
from app.schemas import JobInputType, TailorRequest


RESUME_PARSER_PROMPT = """You are a resume structuring agent.
Given raw resume text, extract strict JSON with keys: name, summary,
skills[], projects[] (title, description, tech_used[], metrics[]),
experience[] (role, company, dates, bullets[]), education[]. Do not
rewrite or improve content - copy facts exactly. Output ONLY valid JSON."""

SOURCE_EXTRACTOR_PROMPT = """Extract technical signal from this source. Return JSON: {source_type, technologies[],
themes[], one_line_summary}. Only include technologies actually used/
discussed, not incidentally mentioned. If this is not real engineering
content (e.g. marketing page), return {'valid': false}."""

SIGNAL_SYNTHESIZER_PROMPT = """Merge these signals from a company's
job posting, engineering blog, and GitHub into one profile. Return JSON:
{confirmed_tech_stack[] (tech appearing in 2+ sources, or strongly in
github/blog even if not in job post - mark others as 'unconfirmed'),
engineering_values[] (max 5, each must cite its source), voice_style
(one line describing how this company talks about engineering)}. Do not
invent anything not present in the inputs."""

MATCH_SCORER_PROMPT = """Compare the candidate's real background
against company_profile.confirmed_tech_stack. For each tech signal,
classify candidate's evidence as 'direct', 'adjacent', or 'none' - never
mark 'direct' unless the candidate's own text explicitly supports it.
Also rank the candidate's projects/experience by relevance to this
company, and select the top 3-4 to foreground. Attach a confidence label
to each ranked item. Return JSON: {matches: [...], selected_projects: [...],
confidence_notes: [...]}."""

CONTENT_GENERATOR_PROMPT = """Rewrite this resume for a specific
company application, and write one tailored cover letter paragraph
(120-180 words). Non-negotiable rules: (1) never add a skill, tool,
metric, or achievement not present in candidate_profile; (2) you may
reorder bullets/projects using match_data.selected_projects; (3) you may
rephrase using company terminology only if it's a genuine synonym for
what the candidate actually did; (4) never upgrade an 'adjacent' match
into 'direct' language; (5) match company_profile.voice_style in the
cover letter, avoid generic phrases like 'perfect fit' or 'passionate
about'. Return JSON: {rewritten_resume: {...same schema as
candidate_profile...}, cover_letter: str, changes: [{edit, reason}]}."""

VERIFIER_PROMPT = """You are a fact-checker, not a writer.
Compare draft.rewritten_resume and draft.cover_letter against
candidate_profile. Flag as FAIL any skill/tool/metric/date/title that
doesn't appear in or match the original, and any 'direct experience'
claim not supported by the original. Also flag any sentence that reads
noticeably more generic/AI-templated than the original's natural tone
(overused words like 'leverage', 'spearheaded', 'synergy'). Be strict -
when in doubt, flag it. Return JSON: {pass: bool, violations: [...],
tone_flags: [...]}."""

RESUME_FORMATTER_PROMPT = """You are a resume formatting agent,
not a content writer. Given this verified resume JSON, do not change
any facts, skills, metrics, or wording already approved by the
verifier. Your only job is to organize it into clean, presentation-
ready sections for a professional single-column resume: order fields
as name/contact, summary, skills, experience, projects, education;
trim any empty sections; ensure bullet points are concise single lines
(max ~160 chars) without truncating meaning; ensure skills are a flat
deduplicated list. Return the same JSON schema as candidate_profile,
reorganized/cleaned, with zero new content."""

OUTPUT_FORMATTER_PROMPT = """Given the changes log and company
signals, write a plain-language explanation (max 6 bullets) of what
changed and why, referencing specific real company signals - no vague
reasons. Return as a markdown bullet list."""

MATCH_OVERLAP_CHECKER_PROMPT = """You are a skills alignment validator.
Compare the candidate's list of technical skills against the technical skills required by the job.
Calculate the percentage of required technical skills that are authentically possessed by the candidate (or where the candidate possesses clear, equivalent expertise).
Return JSON with keys:
- overlap_percentage (0 to 100, float)
- aligned_skills (list of matched skills)
- missing_skills (list of required skills not possessed by the candidate)
- match_status (either 'sufficient' if overlap >= 30, or 'fundamental_mismatch' if overlap < 30)
- message (a short explanation)
Do not invent or assume any skills not explicitly supported by the candidate's profile. Output ONLY valid JSON."""

RESUME_QUALITY_EVALUATOR_PROMPT = """You are an ATS Optimization and Resume Quality Assurance Specialist.
Analyze the tailored resume draft against the target job requirements and company profile.
Evaluate:
1. Keyword integration rate (total required keywords vs keywords integrated).
2. Natural vs forced keyword usage.
3. ATS formatting compatibility and quantifiable achievements.
4. Human readability and flow (must not sound AI-generated, repetitive, or forced).
Generate constructive validation feedback: 3-5 strengths, 3-5 weaknesses, and 3-5 actionable suggestions.
Provide numerical scores (0-100) for ATS compatibility and Human Readability.
Calculate an overall score (weighted: 40% keyword integration, 30% ATS score, 30% readability).
Determine if it passes (overall >= 75, ATS >= 70, readability >= 70, integration rate >= 60%).

Return JSON adhering to this schema:
{
  "passed_validation": bool,
  "overall_score": int,
  "keyword_analysis": {
    "total_keywords_from_job": int,
    "keywords_integrated": int,
    "integration_rate": float,
    "missing_critical_keywords": list[str],
    "naturally_integrated_keywords": list[str],
    "forced_keywords": list[str]
  },
  "phrase_analysis": {
    "phrases_used_correctly": list[str],
    "phrases_used_incorrectly": list[str],
    "missing_important_phrases": list[str]
  },
  "feedback": {
    "strengths": list[str],
    "weaknesses": list[str],
    "suggestions": list[str],
    "ats_score": int,
    "human_readability_score": int
  },
  "ready_for_generation": bool
}
Output ONLY valid JSON."""


class TailorState(TypedDict, total=False):
    request: dict[str, Any]
    session_id: str
    candidate_profile: dict[str, Any]
    source_signals: list[dict[str, Any]]
    source_warnings: list[str]
    sources_checked: list[dict[str, Any]]
    company_profile: dict[str, Any]
    match_data: dict[str, Any]
    draft: dict[str, Any]
    verification: dict[str, Any]
    formatted_resume: dict[str, Any]
    retry_count: int
    agent_trace: list[dict[str, str]]
    final_response: dict[str, Any]
    overlap_check: dict[str, Any]
    validation_report: dict[str, Any]


def _trace(state: TailorState, node_name: str, trace_note: str) -> list[dict[str, str]]:
    return [*state.get("agent_trace", []), {"node_name": node_name, "trace_note": trace_note}]


def _count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _source_summary(signal: dict[str, Any]) -> str:
    tech = signal.get("technologies") or []
    themes = signal.get("themes") or []
    bits: list[str] = []
    if tech:
        bits.append("tech: " + ", ".join(map(str, tech[:5])))
    if themes:
        bits.append("themes: " + ", ".join(map(str, themes[:3])))
    return "; ".join(bits) or signal.get("one_line_summary") or "No clear engineering signal found."


def _normalize_item_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("title") or item.get("role") or item.get("company") or item)
    return str(item)


def _ensure_selected_project_count(match_data: dict[str, Any], candidate_profile: dict[str, Any]) -> dict[str, Any]:
    available = [*_countable(candidate_profile.get("projects")), *_countable(candidate_profile.get("experience"))]
    target = min(4, len(available))
    if len(available) >= 3:
        target = max(3, target)
    selected = match_data.get("selected_projects")
    if not isinstance(selected, list):
        selected = []

    seen = {_normalize_item_name(item) for item in selected}
    for item in available:
        if len(selected) >= target:
            break
        name = _normalize_item_name(item)
        if name not in seen:
            selected.append(item)
            seen.add(name)

    match_data["selected_projects"] = selected[:target]
    return match_data


def _countable(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


async def resume_parser_node(state: TailorState) -> dict[str, Any]:
    candidate_profile = await call_claude_json(
        RESUME_PARSER_PROMPT,
        {"resume_text": state["request"]["resume_text"]},
    )
    note = (
        f"Parsed resume: found {_count(candidate_profile.get('projects'))} projects, "
        f"{_count(candidate_profile.get('skills'))} skills, {_count(candidate_profile.get('experience'))} jobs."
    )
    return {"candidate_profile": candidate_profile, "agent_trace": _trace(state, "resume_parser_node", note)}


async def _extract_signal(source_type: str, content: Any, source: str) -> dict[str, Any]:
    signal = await call_claude_json(
        SOURCE_EXTRACTOR_PROMPT,
        {"source_type": source_type, "source": source, "content": content},
    )
    signal.setdefault("source_type", source_type)
    signal.setdefault("source", source)
    return signal


async def source_extractor_node(state: TailorState) -> dict[str, Any]:
    request = state["request"]
    warnings: list[str] = []
    sources_checked: list[dict[str, Any]] = []

    if request["job_input_type"] == JobInputType.url.value:
        try:
            job_content = await fetch_page_text(request["job_input"])
        except FetchError as exc:
            sources_checked.append(
                {
                    "source_type": "job_posting",
                    "identifier": request["job_input"],
                    "status": "failed",
                    "summary": str(exc),
                }
            )
            raise ValueError(str(exc)) from exc
    else:
        job_content = request["job_input"]

    async def extract_job() -> dict[str, Any] | None:
        signal = await _extract_signal("job_posting", job_content, request["job_input"])
        status = "skipped" if signal.get("valid", True) is False else "fetched"
        sources_checked.append(
            {
                "source_type": "job_posting",
                "identifier": request["job_input"],
                "status": status,
                "summary": "Skipped: not real engineering content." if status == "skipped" else _source_summary(signal),
            }
        )
        return signal

    async def extract_blog(url: str) -> dict[str, Any] | None:
        try:
            text = await fetch_page_text(url)
            signal = await _extract_signal("engineering_blog", text, url)
            status = "skipped" if signal.get("valid", True) is False else "fetched"
            sources_checked.append(
                {
                    "source_type": "blog",
                    "identifier": url,
                    "status": status,
                    "summary": "Skipped: not real engineering content." if status == "skipped" else _source_summary(signal),
                }
            )
            return signal
        except (FetchError, Exception) as exc:
            warnings.append(f"Skipped blog URL {url}: {exc}")
            sources_checked.append(
                {"source_type": "blog", "identifier": url, "status": "failed", "summary": str(exc)}
            )
            return None

    async def extract_github(org: str) -> dict[str, Any] | None:
        try:
            repos = await fetch_github_org_repos(org)
            signal = await _extract_signal("github_org", repos, org)
            status = "skipped" if signal.get("valid", True) is False else "fetched"
            sources_checked.append(
                {
                    "source_type": "github",
                    "identifier": org,
                    "status": status,
                    "summary": "Skipped: no engineering repository signal found." if status == "skipped" else _source_summary(signal),
                }
            )
            return signal
        except (FetchError, Exception) as exc:
            warnings.append(f"Skipped GitHub org {org}: {exc}")
            sources_checked.append(
                {"source_type": "github", "identifier": org, "status": "failed", "summary": str(exc)}
            )
            return None

    jobs = [extract_job()]
    jobs.extend(extract_blog(url) for url in request.get("blog_urls", []))
    if request.get("github_org"):
        jobs.append(extract_github(request["github_org"]))

    results = await asyncio.gather(*jobs)
    signals = [result for result in results if result and result.get("valid", True) is not False]
    note = f"Checked {len(sources_checked)} sources: {len(signals)} yielded usable engineering signals."
    return {
        "source_signals": signals,
        "source_warnings": warnings,
        "sources_checked": sources_checked,
        "agent_trace": _trace(state, "source_extractor_node", note),
    }


async def signal_synthesizer_node(state: TailorState) -> dict[str, Any]:
    company_profile = await call_claude_json(
        SIGNAL_SYNTHESIZER_PROMPT,
        {"source_signals": state["source_signals"]},
    )
    note = (
        f"Synthesized company profile: {_count(company_profile.get('confirmed_tech_stack'))} tech signals "
        f"and {_count(company_profile.get('engineering_values'))} engineering values."
    )
    return {"company_profile": company_profile, "agent_trace": _trace(state, "signal_synthesizer_node", note)}


async def match_scorer_node(state: TailorState) -> dict[str, Any]:
    match_data = await call_claude_json(
        MATCH_SCORER_PROMPT,
        {
            "candidate_profile": state["candidate_profile"],
            "company_profile": state["company_profile"],
        },
    )
    match_data = _ensure_selected_project_count(match_data, state["candidate_profile"])
    note = f"Selected {_count(match_data.get('selected_projects'))} projects or experience items as most relevant."
    return {"match_data": match_data, "agent_trace": _trace(state, "match_scorer_node", note)}


async def content_generator_node(state: TailorState) -> dict[str, Any]:
    retry_count = state.get("retry_count", 0)
    payload: dict[str, Any] = {
        "candidate_profile": state["candidate_profile"],
        "company_profile": state["company_profile"],
        "match_data": state["match_data"],
    }
    if retry_count > 0:
        payload["previous_draft"] = state["draft"]
        payload["fix_instruction"] = "Fix only these issues, keep everything else unchanged"
        payload["violations"] = state.get("verification", {}).get("violations", [])
        payload["tone_flags"] = state.get("verification", {}).get("tone_flags", [])

    draft = await call_claude_json(CONTENT_GENERATOR_PROMPT, payload, max_tokens=6000)
    updates: dict[str, Any] = {"draft": draft}
    attempt_note = "Generated first tailored draft."
    if state.get("verification", {}).get("pass") is False:
        updates["retry_count"] = retry_count + 1
        attempt_note = f"Regenerated draft for fix pass {retry_count + 1} using verifier feedback."
    updates["agent_trace"] = _trace(state, "content_generator_node", attempt_note)
    return updates


async def verifier_node(state: TailorState) -> dict[str, Any]:
    verification = await call_claude_json(
        VERIFIER_PROMPT,
        {
            "candidate_profile": state["candidate_profile"],
            "draft": state["draft"],
        },
    )
    if verification.get("pass"):
        note = "Verifier passed the draft with no unsupported claims."
    elif state.get("retry_count", 0) < 2:
        note = (
            f"Verifier flagged {_count(verification.get('violations'))} violations and "
            f"{_count(verification.get('tone_flags'))} tone flags; sending back for a fix pass."
        )
    else:
        note = (
            f"Verifier still found {_count(verification.get('violations'))} violations after retries; "
            "continuing with failed-after-retries status."
        )
    return {"verification": verification, "agent_trace": _trace(state, "verifier_node", note)}


def route_after_verifier(state: TailorState) -> Literal["content_generator_node", "resume_formatter_node"]:
    if state.get("verification", {}).get("pass") is False and state.get("retry_count", 0) < 2:
        return "content_generator_node"
    return "resume_formatter_node"


async def resume_formatter_node(state: TailorState) -> dict[str, Any]:
    formatted_resume = await call_claude_json(
        RESUME_FORMATTER_PROMPT,
        {"verified_resume": state["draft"].get("rewritten_resume", {})},
        max_tokens=5000,
    )
    note = "Cleaned resume formatting and section order; no content changes requested."
    return {"formatted_resume": formatted_resume, "agent_trace": _trace(state, "resume_formatter_node", note)}


async def match_overlap_checker_node(state: TailorState) -> dict[str, Any]:
    overlap_check = await call_claude_json(
        MATCH_OVERLAP_CHECKER_PROMPT,
        {
            "candidate_skills": state["candidate_profile"].get("skills", []),
            "required_technical_skills": state["company_profile"].get("confirmed_tech_stack", [])
        }
    )
    
    note = f"Technical skill overlap: {overlap_check.get('overlap_percentage', 0)}% - Status: {overlap_check.get('match_status')}"
    
    if overlap_check.get("match_status") == "fundamental_mismatch":
        raise ValueError(
            f"fundamental_mismatch: Technical skill overlap is only {overlap_check.get('overlap_percentage', 0):.1f}%, "
            f"which is below the 30% requirement. Missing skills: {', '.join(overlap_check.get('missing_skills', []))}"
        )
        
    return {
        "overlap_check": overlap_check,
        "agent_trace": _trace(state, "match_overlap_checker_node", note)
    }


async def resume_quality_evaluator_node(state: TailorState) -> dict[str, Any]:
    validation_report = await call_claude_json(
        RESUME_QUALITY_EVALUATOR_PROMPT,
        {
            "job_description": state["request"]["job_input"],
            "company_profile": state["company_profile"],
            "tailored_resume": state["formatted_resume"]
        }
    )
    
    note = f"Evaluated tailored resume quality: Overall Score: {validation_report.get('overall_score', 0)}/100"
    return {
        "validation_report": validation_report,
        "agent_trace": _trace(state, "resume_quality_evaluator_node", note)
    }


async def output_formatter_node(state: TailorState) -> dict[str, Any]:
    diff = await call_claude_text(
        OUTPUT_FORMATTER_PROMPT,
        {
            "changes": state["draft"].get("changes", []),
            "company_profile": state["company_profile"],
            "source_signals": state["source_signals"],
        },
    )
    verification_passed = bool(state.get("verification", {}).get("pass"))
    final_trace = _trace(state, "output_formatter_node", "Formatted final explanation and assembled the API response.")
    final_response = {
        "session_id": state["session_id"],
        "rewritten_resume": state.get("formatted_resume") or state["draft"].get("rewritten_resume", {}),
        "cover_letter": state["draft"].get("cover_letter", ""),
        "diff_explanation": parse_markdown_bullets(diff),
        "verification_status": "passed" if verification_passed else "failed_after_retries",
        "confidence_notes": state.get("match_data", {}).get("confidence_notes", []),
        "sources_checked": state.get("sources_checked", []),
        "agent_trace": final_trace,
        "validation_report": state.get("validation_report")
    }
    return {"final_response": final_response, "agent_trace": final_trace}


def build_graph():
    builder = StateGraph(TailorState)
    builder.add_node("resume_parser_node", resume_parser_node)
    builder.add_node("source_extractor_node", source_extractor_node)
    builder.add_node("signal_synthesizer_node", signal_synthesizer_node)
    builder.add_node("match_overlap_checker_node", match_overlap_checker_node)
    builder.add_node("match_scorer_node", match_scorer_node)
    builder.add_node("content_generator_node", content_generator_node)
    builder.add_node("verifier_node", verifier_node)
    builder.add_node("resume_formatter_node", resume_formatter_node)
    builder.add_node("resume_quality_evaluator_node", resume_quality_evaluator_node)
    builder.add_node("output_formatter_node", output_formatter_node)

    builder.add_edge(START, "resume_parser_node")
    builder.add_edge("resume_parser_node", "source_extractor_node")
    builder.add_edge("source_extractor_node", "signal_synthesizer_node")
    builder.add_edge("signal_synthesizer_node", "match_overlap_checker_node")
    builder.add_edge("match_overlap_checker_node", "match_scorer_node")
    builder.add_edge("match_scorer_node", "content_generator_node")
    builder.add_edge("content_generator_node", "verifier_node")
    builder.add_conditional_edges("verifier_node", route_after_verifier)
    builder.add_edge("resume_formatter_node", "resume_quality_evaluator_node")
    builder.add_edge("resume_quality_evaluator_node", "output_formatter_node")
    builder.add_edge("output_formatter_node", END)
    return builder.compile()


tailor_graph = build_graph()


async def run_tailor_pipeline(request: TailorRequest) -> dict[str, Any]:
    result = await tailor_graph.ainvoke(
        {"request": request.model_dump(), "retry_count": 0, "session_id": str(uuid.uuid4()), "agent_trace": []},
        {"recursion_limit": 25},
    )
    return result["final_response"]

