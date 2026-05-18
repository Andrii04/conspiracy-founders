#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARCTICSHIFT_API_BASE = "https://arctic-shift.photon-reddit.com/api"
ARCTICSHIFT_TIMEOUT = 30
ARCTICSHIFT_MAX_RETRIES = 3
ARCTICSHIFT_RETRY_SLEEP = 2.0

# Rate limiting: 1 request per second
RATE_LIMIT_DELAY = 1.0

USER_AGENT = os.getenv("USER_AGENT", "ArcticShiftFetcher/1.0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Global rate limiter to ensure no more than 1 request per second."""

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait until we're allowed to make another request."""
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time

            if time_since_last < self.delay:
                wait_time = self.delay - time_since_last
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()


# ---------------------------------------------------------------------------
# ArcticShift API
# ---------------------------------------------------------------------------

async def fetch_subreddit_data(
        session: aiohttp.ClientSession,
        subreddit: str,
        rate_limiter: RateLimiter,
) -> Optional[dict]:
    """
    Fetch subreddit data from ArcticShift API.
    Returns the API response as a dict, or None if the request fails.
    """
    url = f"{ARCTICSHIFT_API_BASE}/subreddits/search"
    params = {"subreddit": subreddit}

    for attempt in range(1, ARCTICSHIFT_MAX_RETRIES + 1):
        try:
            await rate_limiter.acquire()
            async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=ARCTICSHIFT_TIMEOUT),
            ) as resp:
                if resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", 60))
                    log.warning(f"Rate limited for r/{subreddit} – sleeping {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status == 404:
                    log.debug(f"r/{subreddit} – Not found (404)")
                    return {
                        "subreddit": subreddit,
                        "error": "not_found",
                        "status_code": 404,
                    }

                if resp.status != 200:
                    log.warning(f"r/{subreddit} – HTTP {resp.status}")
                    if attempt < ARCTICSHIFT_MAX_RETRIES:
                        await asyncio.sleep(ARCTICSHIFT_RETRY_SLEEP * attempt)
                        continue
                    return {
                        "subreddit": subreddit,
                        "error": f"http_{resp.status}",
                        "status_code": resp.status,
                    }

                data = await resp.json()

                # Add the queried subreddit name for reference
                data["_queried_subreddit"] = subreddit

                log.info(f"r/{subreddit} – Successfully fetched")
                return data

        except asyncio.TimeoutError:
            log.warning(f"r/{subreddit} – Timeout (attempt {attempt}/{ARCTICSHIFT_MAX_RETRIES})")
            if attempt < ARCTICSHIFT_MAX_RETRIES:
                await asyncio.sleep(ARCTICSHIFT_RETRY_SLEEP * attempt)

        except aiohttp.ClientError as exc:
            log.warning(f"r/{subreddit} – Client error: {exc} (attempt {attempt}/{ARCTICSHIFT_MAX_RETRIES})")
            if attempt < ARCTICSHIFT_MAX_RETRIES:
                await asyncio.sleep(ARCTICSHIFT_RETRY_SLEEP * attempt)

        except Exception as exc:
            log.error(f"r/{subreddit} – Unexpected error: {exc}")
            return {
                "subreddit": subreddit,
                "error": "unexpected_error",
                "error_message": str(exc),
            }

    # All retries exhausted
    return {
        "subreddit": subreddit,
        "error": "max_retries_exceeded",
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

async def process_subreddit(
        subreddit: str,
        session: aiohttp.ClientSession,
        rate_limiter: RateLimiter,
        semaphore: asyncio.Semaphore,
) -> dict:
    """Process a single subreddit with concurrency control."""
    async with semaphore:
        return await fetch_subreddit_data(session, subreddit, rate_limiter)


async def main(input_file: Path, output_file: Path, concurrency: int):
    """Main processing loop."""

    # Read subreddits from CSV file
    if not input_file.exists():
        log.error(f"Input file not found: {input_file}")
        sys.exit(1)

    subreddits = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            subreddit = line.strip()
            if subreddit and not subreddit.startswith('#'):
                # Remove 'r/' prefix if present
                subreddit = subreddit.lstrip('r/').strip('/')
                if subreddit:
                    subreddits.append(subreddit)

    if not subreddits:
        log.error(f"No valid subreddits found in {input_file}")
        sys.exit(1)

    log.info(f"Processing {len(subreddits)} subreddit(s) from {input_file}")

    # Initialize rate limiter and semaphore
    rate_limiter = RateLimiter(delay=RATE_LIMIT_DELAY)
    semaphore = asyncio.Semaphore(concurrency)

    # Setup HTTP session
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    connector = aiohttp.TCPConnector(limit=32)

    # Open output file for writing
    output_file.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        # Create tasks for all subreddits
        tasks = [
            process_subreddit(sub, session, rate_limiter, semaphore)
            for sub in subreddits
        ]

        # Process with progress bar
        results = []
        with open(output_file, 'w', encoding='utf-8') as f:
            for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Fetching"):
                result = await coro
                results.append(result)

                # Write each result as a JSON line
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # Ensure data is written immediately

    log.info(f"Completed! Results written to {output_file}")
    log.info(f"Total subreddits processed: {len(results)}")

    # Count errors
    errors = sum(1 for r in results if r.get("error"))
    if errors:
        log.warning(f"Encountered errors for {errors} subreddit(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch subreddit data from ArcticShift API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("wa_communities.csv"),
        help="Input CSV file with subreddit names (default: wa_communities.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("../../data/external/arcticshift_data.jsonl"),
        help="Output JSONL file (default: ../../data/external/arcticshift_data.jsonl)",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=5,
        help="Max concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(main(args.input, args.output, args.concurrency))