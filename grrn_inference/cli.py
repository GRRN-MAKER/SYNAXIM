"""
SYNAXIM: Command-Line Interface
=================================
Provides `grrn-convert` and `grrn-serve` commands.

Usage:
    grrn-convert meta-llama/Llama-3.1-8B ./llama-symb --quantize int4
    grrn-serve ./llama-symb --port 8000 --api-key mysecret
"""

from __future__ import annotations

import argparse
import sys


def convert_cli():
    """CLI entry point for model conversion."""
    parser = argparse.ArgumentParser(
        prog="grrn-convert",
        description="Convert a HuggingFace model to GRRN .symb format",
    )
    parser.add_argument("source", help="HuggingFace model ID or local path")
    parser.add_argument("output", help="Output directory for .symb files")
    parser.add_argument(
        "--quantize", choices=["int4", "int8", "fp16", "fp32"],
        default="int4", help="Quantization method (default: int4)"
    )
    parser.add_argument(
        "--architecture", choices=["auto", "dense", "moe"],
        default="auto", help="Model architecture (default: auto-detect)"
    )
    parser.add_argument(
        "--group-size", type=int, default=128,
        help="Quantization group size (default: 128)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    from .export import SymbioticConverter
    converter = SymbioticConverter(verbose=not args.quiet)
    converter.convert(
        source=args.source,
        output_dir=args.output,
        quantize=args.quantize,
        architecture=args.architecture,
        group_size=args.group_size,
    )


def serve_cli():
    """CLI entry point for serving."""
    parser = argparse.ArgumentParser(
        prog="grrn-serve",
        description="Start an OpenAI-compatible inference server",
    )
    parser.add_argument("model", help="Path to .symb model directory")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument("--model-name", default=None, help="Override model name in API responses")

    args = parser.parse_args()

    from .server import GRRNServer
    server = GRRNServer(
        model_path=args.model,
        api_key=args.api_key,
        model_name=args.model_name,
    )
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        sys.argv.pop(1)
        serve_cli()
    else:
        convert_cli()
