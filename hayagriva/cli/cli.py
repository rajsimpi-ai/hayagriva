"""Command line interface for Hayagriva."""
from __future__ import annotations

import argparse
import sys

from hayagriva.core.hayagriva import Hayagriva
from hayagriva.ui.app import launch_ui
from hayagriva.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hayagriva RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch Gradio UI")
    ui_parser.add_argument("--port", type=int, default=7878)
    ui_parser.add_argument("--share", action="store_true", help="Enable public Gradio share")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest text documents")
    ingest_parser.add_argument("paths", nargs="*", help="Paths to text files or directories")

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Question to ask the index")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rag = Hayagriva()

    if args.command == "ui":
        launch_ui(rag, port=args.port, share=args.share)
    elif args.command == "ingest":
        if not args.paths:
            logger.error("No paths provided for ingestion")
            sys.exit(1)
        from hayagriva.ingestion.loaders import load_from_paths

        docs = load_from_paths(args.paths)
        rag.add_documents(docs)
        logger.info("Ingested %d documents", len(docs))
    elif args.command == "query":
        answer = rag.ask(args.question)
        print(answer)
    else:
        # default to showing help
        parse_args(["-h"])


if __name__ == "__main__":
    main()
