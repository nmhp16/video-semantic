# backend/context.py
import json, re, logging
import numpy as np
from collections import Counter
from typing import Optional, List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


def derive_text_summary_for(video_id: str, conn) -> str:
    rows = conn.execute(
        "SELECT text FROM chunks WHERE video_id=? ORDER BY start LIMIT 5", (video_id,)
    ).fetchall()
    if not rows:
        return ""
    texts = [r[0] for r in rows if r[0] and r[0].strip()]
    combined = " ".join(texts)
    return combined[:200] + "..." if len(combined) > 200 else combined


def top_objects_for(video_id: str, conn, k: int = 20) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT objects FROM visual_clips WHERE video_id=?", (video_id,)
    ).fetchall()
    counts: Counter = Counter()
    for (obj_json,) in rows:
        if obj_json:
            try:
                for o in json.loads(obj_json):
                    counts[o] += 1
            except (json.JSONDecodeError, TypeError):
                pass
    return dict(counts.most_common(k))


def top_actions_for(video_id: str, conn, k: int = 20) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT text FROM chunks WHERE video_id=?", (video_id,)
    ).fetchall()
    nlp = _get_nlp()
    verb_counts: Counter = Counter()
    vo_counts: Counter = Counter()
    for (txt,) in rows:
        if not txt:
            continue
        doc = nlp(txt)
        for tok in doc:
            if tok.pos_ == "VERB" and tok.lemma_ not in ("be", "have", "do"):
                v = tok.lemma_.lower()
                verb_counts[v] += 1
                for child in tok.children:
                    if child.dep_ in ("dobj", "obj") and child.pos_ in ("PROPN", "NOUN", "PRON"):
                        vo_counts[f"{v} {child.lemma_.lower()}"] += 1
                        break
    out: Dict[str, int] = {}
    for key, count in vo_counts.most_common(k):
        out[key] = count
    if len(out) < k:
        for verb, count in verb_counts.most_common(k - len(out)):
            out[verb] = count
    return out


def _fetch_texts_for_video(video_id: str, conn, max_chunks: int = 300) -> List[str]:
    rows = conn.execute(
        "SELECT text FROM chunks WHERE video_id=? ORDER BY start LIMIT ?",
        (video_id, max_chunks),
    ).fetchall()
    texts = [t for (t,) in rows if t and t.strip()]
    return [re.sub(r"\s+", " ", t).strip() for t in texts]


def derive_topics(summary: str, objects_topk: dict, actions_topk: dict,
                  texts: list = None, topn: int = 10) -> List[str]:
    texts = texts or []
    corpus = [t for t in texts if t] + ([summary] if summary else [])
    topics: set = set(list(objects_topk.keys())[:5]) | set(list(actions_topk.keys())[:5])
    if corpus:
        vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=1000,
            stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        X = vec.fit_transform(corpus)
        scores = X.sum(axis=0).A1
        vocab = np.array(vec.get_feature_names_out())
        for term in vocab[scores.argsort()[::-1]]:
            if term in topics or re.match(r"^\d+$", term):
                continue
            topics.add(term)
            if len(topics) >= topn + 10:
                break
    return list(topics)


def derive_topics_bertopic(texts: List[str], topn: int = 10,
                           min_topic_size: int = 5) -> List[str]:
    from bertopic import BERTopic
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    topic_model = BERTopic(
        min_topic_size=min_topic_size, calculate_probabilities=False,
        verbose=False, embedding_model="BAAI/bge-small-en-v1.5",
    )
    topics, _ = topic_model.fit_transform(texts)
    info = topic_model.get_topic_info()
    out: List[str] = []
    for _, row in info.sort_values("Count", ascending=False).head(5).iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        for term, _ in (topic_model.get_topic(tid) or [])[:3]:
            out.append(term)
    final, seen = [], set()
    for t in out:
        t = t.strip().lower()
        if t and t not in seen:
            final.append(t); seen.add(t)
        if len(final) >= topn:
            break
    return final


def build_video_context(video_id: str) -> None:
    from db import db
    from embeddings import get_emb
    try:
        with db() as conn:
            summary = derive_text_summary_for(video_id, conn)
            objects_topk = top_objects_for(video_id, conn, k=20)
            actions_topk = top_actions_for(video_id, conn, k=20)
            texts = _fetch_texts_for_video(video_id, conn, max_chunks=300)
            topics = (derive_topics_bertopic(texts, topn=10, min_topic_size=5)
                      if len(texts) >= 10
                      else derive_topics(summary, objects_topk, actions_topk, texts, topn=10))
            fused: List[str] = []
            seen: set = set()
            for t in (list(objects_topk.keys())[:5] + list(actions_topk.keys())[:5] + topics):
                t0 = t.strip().lower()
                if t0 and t0 not in seen:
                    fused.append(t0); seen.add(t0)
            text_for_emb = summary or " ".join(fused)
            emb_blob = (get_emb().encode([text_for_emb], normalize_embeddings=True)
                        .astype("float32")[0].tobytes()
                        if text_for_emb.strip() else None)
            conn.execute("""
                INSERT INTO video_context(video_id,title,source_url,summary,topics,
                                          objects_topk,actions_topk,lang,emb,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(video_id) DO UPDATE SET
                  summary=excluded.summary, topics=excluded.topics,
                  objects_topk=excluded.objects_topk, actions_topk=excluded.actions_topk,
                  emb=excluded.emb, updated_at=CURRENT_TIMESTAMP
            """, (video_id, None, None, summary, json.dumps(fused),
                  json.dumps(objects_topk), json.dumps(actions_topk), "en", emb_blob))
            conn.commit()
        logger.info("Built context for %s: %d topics", video_id, len(fused))
    except Exception:
        logger.exception("Error building video context for %s", video_id)


def _fetch_contexts(video_ids: Optional[List[str]] = None) -> List[Dict]:
    from db import db
    with db() as conn:
        if video_ids:
            qmarks = ",".join(["?"] * len(video_ids))
            rows = conn.execute(
                f"SELECT video_id,topics,objects_topk,actions_topk,emb FROM video_context WHERE video_id IN ({qmarks})",
                video_ids,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT video_id,topics,objects_topk,actions_topk,emb FROM video_context"
            ).fetchall()
    out = []
    for vid, topics, objs, acts, emb in rows:
        out.append({
            "video_id": vid,
            "topics": json.loads(topics or "[]"),
            "objects_topk": json.loads(objs or "{}"),
            "actions_topk": json.loads(acts or "{}"),
            "emb": np.frombuffer(emb, dtype=np.float32) if emb else None,
        })
    return out


def filter_videos_by_context(query: str, restrict_videos: Optional[List[str]] = None,
                              topn: int = 50, min_cos: float = 0.18) -> List[str]:
    from embeddings import get_emb
    qv = get_emb().encode([query], normalize_embeddings=True).astype("float32")[0]
    contexts = _fetch_contexts(restrict_videos)
    scored: List[Tuple[str, float]] = []
    for c in contexts:
        if c["emb"] is None:
            scored.append((c["video_id"], 0.1)); continue
        cos = float(np.dot(qv, c["emb"]))
        q_lower = query.lower()
        if any(t.lower() in q_lower for t in c["topics"]):
            cos += 0.05
        if any(o.lower() in q_lower for o in c["objects_topk"]):
            cos += 0.03
        if any(a.lower() in q_lower for a in c["actions_topk"]):
            cos += 0.03
        scored.append((c["video_id"], cos))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [vid for vid, s in scored if s >= min_cos][:topn]


def passes_hard_context(video_id: str, require_any: Optional[List[str]] = None,
                         require_all: Optional[List[str]] = None) -> bool:
    contexts = _fetch_contexts([video_id])
    if not contexts:
        return True
    c = contexts[0]
    bag = {w.lower() for w in [*c["topics"], *c["objects_topk"], *c["actions_topk"]]}
    if require_all and not all(w.lower() in bag for w in require_all):
        return False
    if require_any and not any(w.lower() in bag for w in require_any):
        return False
    return True
