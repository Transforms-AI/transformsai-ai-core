"""CLI entry point for transformsai-ai-core."""

import argparse
import sys
from pathlib import Path


def config_validate_command(args) -> int:
    """`config validate PATH`: strict by default; exit 1 on errors (the pre-deploy gate)."""
    from transformsai_ai_core.config import ConfigError, load_config, validate_config

    path = Path(args.path)
    if not path.exists():
        print(f"error: config file not found: {path}", file=sys.stderr)
        return 1
    if args.lenient:
        try:
            _, report = load_config(path)
        except ConfigError as exc:
            report = exc.report
    else:
        report = validate_config(path)
    print(report)
    return 0 if report.ok else 1


def config_example_command(args) -> int:
    from transformsai_ai_core.config import generate_example

    text = generate_example()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def config_schema_command(args) -> int:
    from transformsai_ai_core.config import Config

    sys.stdout.write(Config.schema_json(strict=not args.open) + "\n")
    return 0


def download_models_command(args) -> int:
    """Handle 'download models' command."""
    from transformsai_ai_core.central_logger import configure_logging, get_logger
    from transformsai_ai_core.config import ConfigError, load_config

    configure_logging()
    logger = get_logger()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    logger.info(f"Loading config from: {config_path}")
    try:
        config, report = load_config(config_path)
    except ConfigError as exc:
        logger.error(str(exc))
        return 1
    for issue in report.warnings:
        logger.warning(str(issue))

    results = config.download_models(base_url=args.base_url)
    failed = [name for name, path in results.items() if path is None]
    for name, path in results.items():
        if path is not None:
            logger.info(f"Model {name}: {path}")
    if failed:
        logger.error(f"Model download failed: {', '.join(failed)}")
        return 1
    logger.info("Model downloads completed successfully")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="transformsaicore",
        description="TransformsAI Core Library CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # config validate | example | schema
    config_parser = subparsers.add_parser("config", help="Validate configs, generate the example and schema")
    config_sub = config_parser.add_subparsers(dest="action", help="Config action")

    validate_parser = config_sub.add_parser("validate", help="Validate a config (strict; exit 1 on errors)")
    validate_parser.add_argument("path", nargs="?", default="./config.yaml", help="Config file (default: ./config.yaml)")
    validate_parser.add_argument("--lenient", action="store_true",
                                 help="Runtime rules: unknown keys and bad optional values are warnings")

    example_parser = config_sub.add_parser("example", help="Print the commented example config")
    example_parser.add_argument("-o", "--output", help="Write to this file instead of stdout")

    schema_parser = config_sub.add_parser("schema", help="Print the config JSON Schema")
    schema_parser.add_argument("--open", action="store_true", help="Allow unknown keys (default: flag them)")

    # download models
    download_parser = subparsers.add_parser("download", help="Download resources")
    download_subparsers = download_parser.add_subparsers(dest="resource", help="Resource type")
    models_parser = download_subparsers.add_parser("models", help="Download models from config")
    models_parser.add_argument(
        "--config",
        default="./config.yaml",
        help="Path to config file (default: ./config.yaml)",
    )
    models_parser.add_argument("--base-url", default=None, help="Model download API base URL")

    args = parser.parse_args()

    if args.command == "config" and args.action == "validate":
        sys.exit(config_validate_command(args))
    elif args.command == "config" and args.action == "example":
        sys.exit(config_example_command(args))
    elif args.command == "config" and args.action == "schema":
        sys.exit(config_schema_command(args))
    elif args.command == "download" and args.resource == "models":
        sys.exit(download_models_command(args))
    elif args.command == "config":
        config_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
