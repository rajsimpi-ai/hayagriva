"""Minimal usage example for Hayagriva."""
from hayagriva import Hayagriva


if __name__ == "__main__":
    rag = Hayagriva()
    rag.add_documents(["Hayagriva restores knowledge and brings clarity."])
    print(rag.ask("What does Hayagriva do?"))
