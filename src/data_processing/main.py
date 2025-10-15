# main.py
"""
CLI entry point to run different parts of the pipeline.

Examples:
    python main.py --task clean
    python main.py --task sudo
    python main.py --task k2000
    python main.py --task print_examples --model baseline_0 --trigger clean
"""

import argparse
import logging
from pipelines import build_clean_baseline, build_sudo_baseline, build_k2000_baseline, print_some_examples

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Dataset poisoning / evaluation pipeline")
    p.add_argument("--task", type=str, required=True, choices=["clean", "sudo", "k2000", "print_examples"], help="Which pipeline to run")
    p.add_argument("--model", type=str, default="baseline_0", help="Model name (affects save paths)")
    p.add_argument("--trigger", type=str, default="clean", help="Trigger label for some datasets")
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("Running task %s", args.task)

    if args.task == "clean":
        build_clean_baseline(trigger=args.trigger, model_name=args.model)
    elif args.task == "sudo":
        build_sudo_baseline(model_name=args.model)
    elif args.task == "k2000":
        build_k2000_baseline(model_name=args.model, trigger_label=args.trigger)
    elif args.task == "print_examples":
        print_some_examples(model_name=args.model, trigger=args.trigger)
    else:
        raise ValueError("Unknown task: %s" % args.task)


if __name__ == "__main__":
    main()
