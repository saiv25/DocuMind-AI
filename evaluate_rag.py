"""Offline retrieval evaluation with optional Ragas metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

from rag_engine import (  # noqa: E402
    build_messages,
    build_or_load_vector_store,
    retrieve_documents,
)


def normalise(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip().casefold()


def keyword_coverage(
    keywords: list[str],
    text: str,
) -> float:
    clean = [
        normalise(keyword)
        for keyword in keywords
        if normalise(keyword)
    ]

    if not clean:
        return 0.0

    content = normalise(text)

    return (
        sum(1 for term in clean if term in content)
        / len(clean)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate DocuMind's hybrid retrieval."
    )

    parser.add_argument(
        "--pdf-dir",
        default=str(BASE_DIR / "data"),
    )

    parser.add_argument(
        "--dataset",
        default=str(
            BASE_DIR / "evaluation" / "questions.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            BASE_DIR / "evaluation" / "results.csv"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--use-ocr",
        action="store_true",
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
    )

    parser.add_argument(
        "--ragas",
        action="store_true",
        help=(
            "Optional Ragas faithfulness/context-precision scoring. "
            "Requires requirements-eval.txt and reference_answer entries."
        ),
    )

    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)

    files = [
        (path.name, path.read_bytes())
        for path in sorted(pdf_dir.glob("*.pdf"))
    ]

    if not files:
        print(
            f"No PDFs found in {pdf_dir}. "
            "Place evaluation documents there first."
        )
        return 2

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 2

    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    cases = [
        row
        for row in dataset
        if (
            isinstance(row, dict)
            and str(row.get("question", "")).strip()
            and not str(row.get("question", "")).startswith(
                "REPLACE_"
            )
        )
    ]

    if not cases:
        print(
            "No usable questions. Replace the sample placeholders "
            "in evaluation/questions.json."
        )
        return 2

    store, collection, stats = build_or_load_vector_store(
        files,
        workspace_id="offline-evaluation",
        use_ocr=args.use_ocr,
        use_vision=False,
        api_key="",
        vision_model=os.getenv(
            "GROQ_VISION_MODEL",
            "qwen/qwen3.8-27b",
        ),
        persist_directory=BASE_DIR / "chroma_db_evaluation",
    )

    print(
        f"Indexed/reused {stats['chunks']} chunks. "
        f"Collection: {collection}"
    )

    ragas_faithfulness = None
    ragas_context_precision = None
    answer_llm = None

    if args.ragas:
        api_key = os.getenv(
            "GROQ_API_KEY",
            "",
        ).strip()

        if not api_key:
            print(
                "--ragas requires GROQ_API_KEY in .env."
            )
            return 2

        try:
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                ContextPrecision,
                Faithfulness,
            )
            from langchain_groq import ChatGroq

            evaluator_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=90.0,
                max_retries=2,
            )

            eval_model = (
                os.getenv(
                    "GROQ_MODEL",
                    "openai/gpt-oss-20b",
                ).strip()
                or "openai/gpt-oss-20b"
            )

            evaluator_llm = llm_factory(
                eval_model,
                provider="openai",
                client=evaluator_client,
            )

            ragas_faithfulness = Faithfulness(
                llm=evaluator_llm
            )

            ragas_context_precision = ContextPrecision(
                llm=evaluator_llm
            )

            answer_llm = ChatGroq(
                model=eval_model,
                api_key=api_key,
                temperature=0.0,
                max_retries=1,
                timeout=60,
            )

        except Exception as exc:
            print(
                "Could not initialise optional Ragas scoring. "
                "Install requirements-eval.txt and verify the Groq model. "
                f"Details: {exc}"
            )
            return 2

    results: list[dict[str, Any]] = []

    for number, case in enumerate(cases, start=1):
        question = str(case["question"]).strip()

        expected_keywords = case.get(
            "expected_keywords",
            [],
        )

        if not isinstance(expected_keywords, list):
            expected_keywords = []

        start = time.perf_counter()

        retrieved = retrieve_documents(
            question,
            store,
            top_k=max(1, min(args.top_k, 12)),
            use_reranker=args.rerank,
            hybrid=True,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        chunks = [
            row["document"]
            for row in retrieved
        ]

        context = "\n\n".join(
            doc.page_content
            for doc in chunks
        )

        coverage = keyword_coverage(
            expected_keywords,
            context,
        )

        expected_source = normalise(
            case.get("expected_source", "")
        )

        expected_page = str(
            case.get("expected_page", "")
        ).strip()

        source_rank = ""
        source_page_hit = ""
        keyword_rank = ""

        for rank, doc in enumerate(chunks, start=1):
            source = normalise(
                doc.metadata.get("source", "")
            )

            page = str(
                doc.metadata.get("page", "")
            ).strip()

            page_coverage = keyword_coverage(
                expected_keywords,
                doc.page_content,
            )

            if (
                keyword_rank == ""
                and expected_keywords
                and page_coverage > 0
            ):
                keyword_rank = rank

            if (
                expected_source
                and expected_source in source
                and source_rank == ""
            ):
                source_rank = rank

            if (
                expected_source
                and expected_source in source
                and expected_page
                and page == expected_page
            ):
                source_page_hit = 1

        faithfulness_score = ""
        context_precision_score = ""
        ragas_error = ""

        if (
            args.ragas
            and answer_llm is not None
            and ragas_faithfulness is not None
        ):
            try:
                answer_result = answer_llm.invoke(
                    build_messages(
                        question,
                        [],
                        retrieved,
                    )
                )

                answer = str(
                    getattr(
                        answer_result,
                        "content",
                        answer_result,
                    ) or ""
                ).strip()

                context_list = [
                    doc.page_content
                    for doc in chunks
                ]

                if answer:
                    try:
                        score_result = ragas_faithfulness.score(
                            user_input=question,
                            response=answer,
                            retrieved_contexts=context_list,
                        )

                        faithfulness_score = float(
                            getattr(
                                score_result,
                                "value",
                                score_result,
                            )
                        )

                    except Exception as exc:
                        ragas_error = (
                            f"Faithfulness scoring failed: {exc}"
                        )

                    reference_answer = str(
                        case.get("reference_answer", "")
                    ).strip()

                    if (
                        reference_answer
                        and ragas_context_precision is not None
                    ):
                        try:
                            precision_result = (
                                ragas_context_precision.score(
                                    user_input=question,
                                    reference=reference_answer,
                                    retrieved_contexts=context_list,
                                )
                            )

                            context_precision_score = float(
                                getattr(
                                    precision_result,
                                    "value",
                                    precision_result,
                                )
                            )

                        except Exception as exc:
                            ragas_error = (
                                ragas_error + " | "
                                if ragas_error
                                else ""
                            ) + f"Context precision failed: {exc}"

                else:
                    ragas_error = (
                        "Answer model returned an empty response."
                    )

            except Exception as exc:
                ragas_error = str(exc)[:500]

        results.append({
            "question": question,
            "retrieval_latency_ms": round(
                latency_ms,
                2,
            ),
            "retrieved_chunks": len(chunks),
            "keyword_coverage": round(
                coverage,
                4,
            ),
            "keyword_hit_at_k": int(
                coverage > 0
            ),
            "first_keyword_hit_rank": keyword_rank,
            "keyword_mrr": (
                round(1 / int(keyword_rank), 4)
                if keyword_rank
                else 0.0
            ),
            "expected_source": case.get(
                "expected_source",
                "",
            ),
            "expected_source_rank": source_rank,
            "expected_source_page_hit": source_page_hit,
            "top_source_files": "; ".join(
                str(doc.metadata.get("source", ""))
                for doc in chunks[:3]
            ),
            "ragas_faithfulness": faithfulness_score,
            "ragas_context_precision": context_precision_score,
            "ragas_error": ragas_error,
        })

        print(
            f"[{number}/{len(cases)}] "
            f"hit={int(coverage > 0)} "
            f"coverage={coverage:.2f} "
            f"time={latency_ms:.0f}ms — "
            f"{question[:70]}"
        )

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n--- Retrieval summary ---")

    print(f"Questions: {len(results)}")

    print(
        f"Keyword hit@{args.top_k}: "
        f"{statistics.mean(r['keyword_hit_at_k'] for r in results):.3f}"
    )

    print(
        "Mean keyword coverage: "
        f"{statistics.mean(r['keyword_coverage'] for r in results):.3f}"
    )

    print(
        "Keyword MRR: "
        f"{statistics.mean(r['keyword_mrr'] for r in results):.3f}"
    )

    print(
        "Mean retrieval latency: "
        f"{statistics.mean(r['retrieval_latency_ms'] for r in results):.1f} ms"
    )

    print(f"Detailed results: {out}")

    if args.ragas:
        valid_faith = [
            float(row["ragas_faithfulness"])
            for row in results
            if row["ragas_faithfulness"] != ""
        ]

        valid_precision = [
            float(row["ragas_context_precision"])
            for row in results
            if row["ragas_context_precision"] != ""
        ]

        if valid_faith:
            print(
                "Mean Ragas faithfulness: "
                f"{statistics.mean(valid_faith):.3f}"
            )

        if valid_precision:
            print(
                "Mean Ragas context precision: "
                f"{statistics.mean(valid_precision):.3f}"
            )

        print(
            "Ragas scores depend on the judge model and your "
            "reference answers; inspect failed rows in the CSV."
        )

    print(
        "Keyword-based metrics are diagnostics, not proof of "
        "semantic correctness. Manually review answers and source pages."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())