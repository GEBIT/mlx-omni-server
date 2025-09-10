import argparse
import os

import uvicorn
from fastapi import FastAPI

from mlx_omni_server.utils.logger import set_logger_level

from .middleware.logging import RequestResponseLoggingMiddleware
from .routers import api_router
from .utils.logger import logger, set_logger_level

app = FastAPI(title="MLX Omni Server")

# Add request/response logging middleware with custom levels
app.add_middleware(
    RequestResponseLoggingMiddleware,
    # exclude_paths=["/health"]
)

from fastapi.middleware.cors import CORSMiddleware

app.include_router(api_router)


def build_parser():
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(description="MLX Omni Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to, defaults to 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10240,
        help="Port to bind the server to, defaults to 10240",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use, defaults to 1",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level, defaults to info",
    )

    parser.add_argument(
        "--cors-allow-origins",
        type=str,
        default=None,
        help="Apply origins to CORSMiddleware. This is useful for accessing the local server directly from the browser (use --cors-allow-origins=\"*\"). Defaults to disabled",
    )
    return parser


parser = build_parser()
args = parser.parse_args()

if args.cors_allow_origins:
    origins = [origin.strip() for origin in args.cors_allow_origins.split(",")]

    # Allow all origins for local use or specific origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Set log level through environment variable
os.environ["MLX_OMNI_LOG_LEVEL"] = args.log_level
set_logger_level(logger, args.log_level)


def start():
    """Start the MLX Omni Server."""

    # Start server with uvicorn
    uvicorn.run(
        "mlx_omni_server.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
    )


if __name__ == "__main__":
    start()
