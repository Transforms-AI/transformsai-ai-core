"""CLI entry point for transformsai-ai-core."""

import argparse
import sys
from pathlib import Path


def download_models_command(args):
    """Handle 'download models' command."""
    from transformsai_ai_core.config_loader import process_config
    from transformsai_ai_core.central_logger import get_logger

    logger = get_logger()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading config from: {config_path}")
    base_dir = config_path.parent

    # Process config with model downloading enabled
    try:
        config = process_config(
            config_path=config_path,
            base_dir=base_dir,
            resolve_models=True,
            download_models=True,
        )
        logger.info("Model downloads completed successfully")
    except Exception as e:
        logger.error(f"Error processing config: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="transformsaicore",
        description="TransformsAI Core Library CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download resources")
    download_subparsers = download_parser.add_subparsers(dest="resource", help="Resource type")

    # Download models subcommand
    models_parser = download_subparsers.add_parser("models", help="Download models from config")
    models_parser.add_argument(
        "--config",
        default="./config.yaml",
        help="Path to config file (default: ./config.yaml)",
    )

    args = parser.parse_args()

    if args.command == "download" and args.resource == "models":
        download_models_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
