#!/usr/bin/env python3

import asyncio
import json
import os
import random
import re
from pathlib import Path

from autojudge_base import LeaderboardBuilder, LeaderboardSpec, MeasureSpec, NuggetBanks
from autojudge_base.nugget_data import NuggetBank, NuggetQuestion
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm


MAVESUP_SPEC = LeaderboardSpec(measures=( MeasureSpec("NUGGET_RECALL", description="Bank questions answered on a supported sentence / bank size, 0-1."), MeasureSpec("CITATION_PRECISION", description="Fully supported cited sentences / cited sentences, 0-1."), MeasureSpec("SCORE", description="Harmonic mean of NUGGET_RECALL and CITATION_PRECISION, 0-1."), ))


def _req(request_id, system, user, max_tokens):
    extra = None
    if "gpt-oss" in (os.environ.get("OPENAI_MODEL") or "").lower():
        max_tokens = max(int(max_tokens) * 3, 1500)
        extra = {"reasoning_effort": "low"}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return MinimaLlmRequest(request_id=request_id, messages=messages, temperature=0.0, max_tokens=max_tokens, extra=extra)


async def _run_batched(llm_config, requests):
    raw = llm_config.raw
    backend = OpenAIMinimaLlm(MinimaLlmConfig.from_dict(raw) if raw else MinimaLlmConfig.from_env())
    try:
        return await backend.run_batched(requests)
    finally:
        await backend.aclose()


def _truncated(result):
    try:
        return result.raw["choices"][0]["finish_reason"] == "length"
    except (AttributeError, KeyError, IndexError, TypeError):
        return False


def _run(llm_config, requests):
    requests = list(requests)
    results = asyncio.run(_run_batched(llm_config, requests)) if requests else []
    for _ in range(2):
        idx = [i for i, r in enumerate(results) if _truncated(r)]
        if not idx:
            break
        print(f"[mavesup] {len(idx)} replies cut off by the completion limit, retrying with twice the budget")
        for i in idx:
            r = requests[i]
            requests[i] = MinimaLlmRequest(request_id=r.request_id, messages=r.messages, temperature=0.0, max_tokens=r.max_tokens * 2, extra=r.extra)
        for i, r in zip(idx, asyncio.run(_run_batched(llm_config, [requests[i] for i in idx]))):
            results[i] = r
    return results


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)

def _repair_json(text):
    out = []
    stack = []
    in_str = False
    esc = False
    pair = {"{": "}", "[": "]"}
    for ch in text:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            want = pair[stack[-1]]
            stack.pop()
            out.append(want)
        else:
            out.append(ch)
    if in_str:
        out.append('"')
    while stack:
        out.append(pair[stack.pop()])
    fixed = "".join(out)
    return re.sub(r",\s*([}\]])", r"\1", fixed)


def _parse_json(result):
    if not isinstance(result, MinimaLlmResponse):
        return None
    text = _CODE_FENCE_RE.sub("", result.text.strip())
    start = text.find("{")
    if start < 0:
        return None
    end = max(text.rfind("}"), text.rfind("]"))
    candidate = text[start:end + 1] if end > start else text[start:]
    for attempt in (candidate, _repair_json(candidate)):
        try:
            obj = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


_WORD_RE = re.compile(r"[A-Za-z0-9]{4,}")

def _request_text(topic):
    parts = []
    if topic.title:
        parts.append(f"Title: {topic.title}")
    if topic.background:
        parts.append(f"Background: {topic.background}")
    if topic.problem_statement:
        parts.append(f"Problem statement: {topic.problem_statement}")
    return "\n".join(parts) if parts else f"Request id: {topic.request_id}"


def _nugget_line(nugget):
    question = getattr(nugget, "question", None) or ""
    answers = getattr(nugget, "answers", None) or {}
    texts = [a.answer for a in (answers.values() if isinstance(answers, dict) else answers) if getattr(a, "answer", None)]
    if texts:
        return f"{question} (acceptable answers: {'; '.join(texts[:6])})"
    return question


def _answered_by(obj, n_nuggets, n_sentences):
    rows = obj.get("sentences") if obj else None
    if not isinstance(rows, list):
        return None
    answered = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        s = row.get("sentence")
        if not (isinstance(s, int) and 1 <= s <= n_sentences):
            continue
        for n in row.get("nuggets") or []:
            if isinstance(n, int) and 1 <= n <= n_nuggets and (s - 1) not in answered.setdefault(n, []):
                answered[n].append(s - 1)
    return answered


def _to_bank(proposed, order):
    if proposed is None:
        return None
    return {order[n - 1] + 1: ss for n, ss in proposed.items()}


def _best_excerpt(sentence, doc_text, excerpt_chars, max_doc_chars):
    doc_text = doc_text[:max_doc_chars]
    if len(doc_text) <= excerpt_chars:
        return doc_text
    words = {w.lower() for w in _WORD_RE.findall(sentence)}
    step = max(excerpt_chars // 2, 1)
    best_start, best_score = 0, -1
    for start in range(0, len(doc_text) - excerpt_chars + 1, step):
        window = doc_text[start:start + excerpt_chars]
        score = sum(1 for w in set(_WORD_RE.findall(window.lower())) if w in words)
        if score > best_score:
            best_start, best_score = start, score
    return doc_text[best_start:best_start + excerpt_chars]


NUGGET_SYSTEM = "You write evaluation checklists for report requests. You answer with JSON only."

NUGGET_USER = """A system will write a report in response to the request below. Before reading any report, list the nugget questions a complete and correct report must answer.

Rules:
- Each question asks for one specific fact, date, name, number, mechanism, or position that the request calls for.
- Cover every part of the request, including sub-questions and named entities.
- Do not ask about writing quality, length, or citations.
- Write between {min_nuggets} and {max_nuggets} questions. Mark each "vital" if a report that omits it has failed the request, otherwise "okay".

Request:
{request}

Answer with JSON of the form {{"nuggets": [{{"question": "...", "importance": "vital"}}, ...]}}"""


MATCH_SYSTEM = "You are an assessor checking which required facts a report states. You answer with JSON only."

MATCH_USER = """Request:
{request}

Nugget questions a complete report must answer (numbered):
{nuggets}

Report sentences (index, then text):
{sentences}

Go through the report sentence by sentence, as an assessor reading it. For each sentence, list the numbers of the nugget questions it answers: the sentence states an answer to the question, one of the acceptable answers where they are listed or a plain restatement of one. A sentence that merely touches the subject of a question does not answer it. Use only what the sentence says; do not use your own knowledge to fill gaps. A question may be answered by several sentences; list it under each of them. List only the sentences that answer something.

Answer with JSON of the form {{"sentences": [{{"sentence": 4, "nuggets": [2, 7]}}, {{"sentence": 9, "nuggets": [1]}}, ...]}} or {{"sentences": []}}."""


VERIFY_SYSTEM = "You are an assessor checking whether a report sentence answers a required question. You answer with JSON only."

VERIFY_USER = """Check each item on its own.

{items}

For each item, decide whether the sentence STATES an answer to the question. It must give the answer itself: one of the acceptable answers where they are listed, or a plain restatement of one. A sentence that mentions the subject of the question, promises the answer later, gives background, or answers a different question about the same topic does not count. Use only the sentence itself; do not use the rest of the report or your own knowledge.

Answer with JSON of the form {{"verdicts": [{{"item": 1, "answers": true}}, {{"item": 2, "answers": false}}, ...]}} with one entry per item."""


SUPPORT_SYSTEM = "You are an assessor checking whether cited passages support report sentences. You answer with JSON only."

SUPPORT_RULES = """For each item, first split the sentence into its factual claims: each specific thing it asserts (a fact, a figure, a date, a name, an event, a cause, a position). Connective phrasing and generalities are not claims. Then check each claim against the passage alone. A claim is found ONLY when the passage states it: the passage says the same thing, with the same entities, numbers and dates. A claim is NOT found when the passage is about the same subject but does not state that claim, when it states something weaker or more general, when it names a different entity, figure or date, or when reaching the claim needs a further step of reasoning or world knowledge. If you are unsure, the claim is not found. Then grade the sentence:
- "full": every claim is found.
- "partial": some claims are found, at least one is not.
- "none": no claim is found, or the passage contradicts the sentence.
Judge only from the passage. A claim that is true in the world but absent from the passage is not found."""

SUPPORT_USER = """{rules}

Items:
{items}

Answer with JSON of the form {{"verdicts": [{{"item": 1, "claims": [{{"claim": "...", "found": true}}, {{"claim": "...", "found": false}}], "support": "partial"}}, ...]}} with one entry per item."""


def _fallback_bank(topic):
    topic_id = topic.request_id
    gist = (topic.problem_statement or topic.title or topic_id).strip()[:300]
    fallback = NuggetQuestion.from_lazy(query_id=topic_id, question=f"Does the report address the request: {gist}")
    fallback.metadata = {"fallback": True}
    bank = NuggetBank(query_id=topic_id, title_query=topic.title or topic_id)
    bank.add_nuggets([fallback])
    return bank


class MaVeSupNuggetCreator:

    nugget_banks_type = NuggetBanks

    def create_nuggets(self, rag_topics, llm_config, nugget_banks=None, min_nuggets=8, max_nuggets=15, max_tokens=1500, **kwargs):
        if nugget_banks and nugget_banks.banks:
            kept = []
            for topic in rag_topics:
                bank = nugget_banks.banks.get(topic.request_id)
                kept.append(bank if bank and bank.nuggets_as_list() else _fallback_bank(topic))
            print(f"[mavesup] nuggets: using input banks for {len(kept)} topics ({len(nugget_banks.banks)} supplied), unchanged")
            return NuggetBanks.from_banks_list(kept)
        topics = sorted(rag_topics, key=lambda t: t.request_id)
        requests = [_req(f"nuggets:{t.request_id}", NUGGET_SYSTEM, NUGGET_USER.format(min_nuggets=min_nuggets, max_nuggets=max_nuggets, request=_request_text(t)), max_tokens) for t in topics]
        results = _run(llm_config, requests)

        banks = []
        failed = 0
        for topic, result in zip(topics, results):
            obj = _parse_json(result)
            items = obj.get("nuggets") if obj else None
            questions = []
            for item in items or []:
                q = item.get("question") if isinstance(item, dict) else None
                if not q or not isinstance(q, str):
                    continue
                questions.append(NuggetQuestion.from_lazy(query_id=topic.request_id, question=q.strip()))
            if questions:
                bank = NuggetBank(query_id=topic.request_id, title_query=topic.title or topic.request_id)
                bank.add_nuggets(questions)
            else:
                failed += 1
                bank = _fallback_bank(topic)
            banks.append(bank)

        print(f"[mavesup] nuggets: {len(banks)} topics, {failed} without a usable nugget list")
        return NuggetBanks.from_banks_list(banks)


class MaVeSupJudge:

    def judge(self, rag_responses, rag_topics, llm_config, nugget_banks=None, sentences_per_call=8, excerpt_chars=700, max_doc_chars=20000, match_max_tokens=1200, verify_per_call=10, verify_max_tokens=600, support_max_tokens=1200, permutations=3, on_missing_evals="fix_aggregate", filebase="default", outdir=Path("."), **kwargs):
        topics = {t.request_id: t for t in rag_topics}
        expected_topic_ids = [t.request_id for t in rag_topics]
        reports = sorted(rag_responses, key=lambda r: (r.metadata.run_id, r.metadata.topic_id))
        permutations = max(1, int(permutations))

        builder = LeaderboardBuilder(MAVESUP_SPEC)
        judgments = []
        parse_failures = 0
        match_failures = 0
        perm_measures = {}

        for perm in range(permutations):
            rng = random.Random(perm) if perm > 0 else None
            print(f"[mavesup] permutation {perm + 1}/{permutations}")

            match_requests = []
            support_requests = []
            plan = []
            for report in reports:
                run_id, topic_id = report.metadata.run_id, report.metadata.topic_id
                topic = topics.get(topic_id)
                sentences = report.get_sentences_with_citations() if report.responses else []
                documents = report.documents or {}
                bank = nugget_banks.banks.get(topic_id) if nugget_banks else None
                nuggets = [_nugget_line(n) for n in bank.nuggets_as_list()] if bank else []
                entry = {"run_id": run_id, "topic_id": topic_id, "n_sentences": len(sentences), "n_nuggets": len(nuggets), "match_idx": None, "support": [], "verify": [], "nugget_lines": nuggets, "texts": [s.text for s in sentences], "cited": [], "missing_doc": [], "order": list(range(len(nuggets)))}
                if rng is not None:
                    rng.shuffle(entry["order"])

                if sentences and nuggets and topic is not None:
                    entry["match_idx"] = len(match_requests)
                    prompt = MATCH_USER.format(request=_request_text(topic), nuggets="\n".join(f"{i + 1}. {nuggets[j]}" for i, j in enumerate(entry["order"])), sentences="\n".join(f"[{i + 1}] {s.text}" for i, s in enumerate(sentences)))
                    match_requests.append(_req(f"match:{run_id}:{topic_id}", MATCH_SYSTEM, prompt, match_max_tokens))

                items = []
                for i, s in enumerate(sentences):
                    if not s.citations:
                        continue
                    entry["cited"].append(i)
                    doc = documents.get(s.citations[0])
                    if doc is None or not doc.text:
                        entry["missing_doc"].append(i)
                        continue
                    excerpt = _best_excerpt(s.text, doc.get_text(), excerpt_chars, max_doc_chars)
                    items.append((i, f"Sentence: {s.text}\nPassage: {excerpt}"))
                if rng is not None:
                    rng.shuffle(items)
                for b in range(0, len(items), sentences_per_call):
                    chunk = items[b:b + sentences_per_call]
                    entry["support"].append((len(support_requests), [i for i, _ in chunk]))
                    prompt = SUPPORT_USER.format(rules=SUPPORT_RULES, items="\n\n".join(f"Item {k + 1}\n{text}" for k, (_, text) in enumerate(chunk)))
                    support_requests.append(_req(f"support:{run_id}:{topic_id}:{b // sentences_per_call}", SUPPORT_SYSTEM, prompt, support_max_tokens))
                plan.append(entry)

            print(f"[mavesup] {len(reports)} reports -> {len(match_requests)} match calls, {len(support_requests)} support calls")
            match_results = _run(llm_config, match_requests)
            support_results = _run(llm_config, support_requests)

            verify_requests = []
            pairs_total = 0
            for entry in plan:
                if entry["match_idx"] is None:
                    continue
                entry["proposed"] = _to_bank(_answered_by(_parse_json(match_results[entry["match_idx"]]), entry["n_nuggets"], entry["n_sentences"]), entry["order"])
                if not entry["proposed"]:
                    continue
                pairs = [(n, si) for n, ss in sorted(entry["proposed"].items()) for si in sorted(ss)]
                if rng is not None:
                    rng.shuffle(pairs)
                pairs_total += len(pairs)
                lines, texts = entry["nugget_lines"], entry["texts"]
                for b in range(0, len(pairs), verify_per_call):
                    chunk = pairs[b:b + verify_per_call]
                    entry["verify"].append((len(verify_requests), chunk))
                    prompt = VERIFY_USER.format(items="\n\n".join(f"Item {k + 1}\nQuestion: {lines[n - 1]}\nSentence: {texts[si]}" for k, (n, si) in enumerate(chunk)))
                    verify_requests.append(_req(f"verify:{entry['run_id']}:{entry['topic_id']}:{b // verify_per_call}", VERIFY_SYSTEM, prompt, verify_max_tokens))
            print(f"[mavesup] verification: {pairs_total} proposed matches -> {len(verify_requests)} calls")
            verify_results = _run(llm_config, verify_requests)

            for entry in plan:
                n_nuggets = entry["n_nuggets"]

                grade_of = {i: "none" for i in entry["cited"]}
                for req_idx, sent_indices in entry["support"]:
                    obj = _parse_json(support_results[req_idx])
                    verdicts = obj.get("verdicts") if obj else None
                    if not isinstance(verdicts, list):
                        parse_failures += 1
                        continue
                    for v in verdicts:
                        if not isinstance(v, dict):
                            continue
                        k = v.get("item")
                        if not isinstance(k, int) or not (1 <= k <= len(sent_indices)):
                            continue
                        grade = str(v.get("support", "")).lower()
                        found = [c.get("found") for c in (v.get("claims") or []) if isinstance(c, dict) and "found" in c]
                        if found:
                            grade = "full" if all(found) else "partial" if any(found) else "none"
                        grade_of[sent_indices[k - 1]] = grade if grade in ("full", "partial") else "none"

                answered_by = {}
                if entry["match_idx"] is not None:
                    if entry["proposed"] is None:
                        match_failures += 1
                    else:
                        for req_idx, chunk in entry["verify"]:
                            obj = _parse_json(verify_results[req_idx])
                            verdicts = obj.get("verdicts") if obj else None
                            if not isinstance(verdicts, list):
                                parse_failures += 1
                                continue
                            kept = {v.get("item"): bool(v.get("answers")) for v in verdicts if isinstance(v, dict) and isinstance(v.get("item"), int)}
                            for k, (nug, si) in enumerate(chunk, start=1):
                                if kept.get(k) and si not in answered_by.setdefault(nug, []):
                                    answered_by[nug].append(si)

                hit = {n for n, ss in answered_by.items() if any(grade_of.get(s) in ("full", "partial") for s in ss)}
                nugget_recall = len(hit) / n_nuggets if n_nuggets else 0.0
                citation_precision = sum(g == "full" for g in grade_of.values()) / len(grade_of) if grade_of else 0.0
                perm_measures.setdefault((entry["run_id"], entry["topic_id"]), []).append((nugget_recall, citation_precision))
                if perm > 0:
                    continue
                sentences_out = [{"i": i, "grade": grade_of.get(i), "nuggets": sorted(n for n, ss in answered_by.items() if i in ss)} for i in range(entry["n_sentences"])]
                judgments.append({"run_id": entry["run_id"], "topic_id": entry["topic_id"], "n_sentences": entry["n_sentences"], "n_cited": len(entry["cited"]), "n_missing_doc": len(entry["missing_doc"]), "n_nuggets": n_nuggets, "n_answered": len(answered_by), "n_recalled": len(hit), "sentences": sentences_out})

        for (run_id, topic_id), ms in perm_measures.items():
            nugget_recall = sum(m[0] for m in ms) / len(ms)
            citation_precision = sum(m[1] for m in ms) / len(ms)
            score = 2 * nugget_recall * citation_precision / (nugget_recall + citation_precision) if nugget_recall + citation_precision > 0 else 0.0
            builder.add(run_id=run_id, topic_id=topic_id, values={"NUGGET_RECALL": nugget_recall, "CITATION_PRECISION": citation_precision, "SCORE": score})
        for j in judgments:
            ms = perm_measures[(j["run_id"], j["topic_id"])]
            j["permutations"] = [[round(a, 4), round(b_, 4)] for a, b_ in ms]
            j["nugget_recall"] = round(sum(m[0] for m in ms) / len(ms), 4)
            j["citation_precision"] = round(sum(m[1] for m in ms) / len(ms), 4)

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        judgment_path = outdir / f"{Path(filebase).name}.judgment.jsonl"
        with open(judgment_path, "w", encoding="utf-8") as f:
            for j in judgments:
                f.write(json.dumps(j) + "\n")
        print(f"[mavesup] judgments (ids and counts only) -> {judgment_path}; {parse_failures} unparseable replies, {match_failures} reports without a usable match list")

        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing=on_missing_evals)
        leaderboard.verify(expected_topic_ids=expected_topic_ids, warn=True, on_missing=on_missing_evals)
        return leaderboard
