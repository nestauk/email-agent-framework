"""Evaluate email triage classification using LangSmith."""

import os
from datetime import datetime

import matplotlib.pyplot as plt
from langgraph.store.memory import InMemoryStore
from langsmith import Client

from ..graph import overall_workflow
from .email_dataset import examples_triage

# Compile graph with in-memory store for evaluation
_eval_store = InMemoryStore()
email_assistant = overall_workflow.compile(store=_eval_store)

# Dataset name
DATASET_NAME = "Nesta Email Triage Eval"


def target_email_assistant(inputs: dict) -> dict:
    """Process an email through the workflow-based email assistant.

    Args:
        inputs: A dictionary containing the email_input field from the dataset

    Returns:
        A formatted dictionary with the assistant's response messages
    """
    import traceback

    try:
        response = email_assistant.invoke({"email_input": inputs["email_input"]})
        if "classification_decision" in response:
            return {"classification_decision": response["classification_decision"]}
        else:
            print("No classification_decision in response from workflow agent")
            return {"classification_decision": "unknown"}
    except Exception as e:
        print(f"Error in workflow agent: {e}")
        traceback.print_exc()
        return {"classification_decision": "unknown"}


def classification_evaluator(outputs: dict, reference_outputs: dict) -> bool:
    """Check if the answer exactly matches the expected answer."""
    return outputs["classification_decision"].lower() == reference_outputs["classification"].lower()


def run_evaluation(recreate_dataset: bool = False) -> None:
    """Run the triage evaluation and generate visualization.

    Args:
        recreate_dataset: If True, delete and recreate the dataset even if it exists.
    """
    client = Client()

    # Delete existing dataset if recreate is requested
    if recreate_dataset and client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Deleting existing dataset: {DATASET_NAME}")
        existing = client.read_dataset(dataset_name=DATASET_NAME)
        client.delete_dataset(dataset_id=existing.id)

    # Create the dataset if it doesn't exist
    if not client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Creating dataset: {DATASET_NAME}")
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="A dataset of e-mails and their triage decisions.",
        )
        client.create_examples(dataset_id=dataset.id, examples=examples_triage)
    else:
        print(f"Using existing dataset: {DATASET_NAME}")

    # Run evaluation
    print("Running evaluation...")
    experiment_results = client.evaluate(
        target_email_assistant,
        data=DATASET_NAME,
        evaluators=[classification_evaluator],
        experiment_prefix="E-mail assistant workflow",
        max_concurrency=2,
    )

    # Convert evaluation results to pandas dataframe
    df = experiment_results.to_pandas()

    # Calculate mean score
    score = df["feedback.classification_evaluator"].mean() if "feedback.classification_evaluator" in df.columns else 0.0

    # Create visualization
    plt.figure(figsize=(10, 6))
    models = ["Agentic Workflow"]
    scores = [score]

    plt.bar(models, scores, color=["#5DA5DA"], width=0.5)
    plt.xlabel("Agent Type")
    plt.ylabel("Average Score")
    plt.title("Email Triage Performance - Classification Score")

    for i, s in enumerate(scores):
        plt.text(i, s + 0.02, f"{s:.2f}", ha="center", fontweight="bold")

    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Ensure output directory exists
    output_dir = "outputs/eval"
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{output_dir}/triage_evaluation_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()

    print("\nEvaluation complete!")
    print(f"Score: {score:.2f}")
    print(f"Visualization saved to: {plot_path}")


def main() -> None:
    """CLI entry point for triage evaluation."""
    import sys

    recreate = "--recreate-dataset" in sys.argv
    run_evaluation(recreate_dataset=recreate)


if __name__ == "__main__":
    main()
