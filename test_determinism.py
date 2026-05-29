#!/usr/bin/env python3
"""Test output determinism of a local vLLM server.

Sends the same prompt many times (concurrently, to exercise the server's
dynamic batching) and reports how many *unique* answers came back. With
batch-invariant kernels + temperature 0, the answer should be 1.

Usage:
    uv run python test_determinism.py
    uv run python test_determinism.py --n 100 --concurrency 16
"""
import argparse
import hashlib
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

PROMPT = "What do you know about a singer Johnny Cash? Briefly."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--n", type=int, default=50, help="number of requests")
    p.add_argument("--concurrency", type=int, default=8,
                   help="parallel in-flight requests (stresses batching)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-answers", action="store_true",
                   help="print the full text of every unique answer")
    return p.parse_args()


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    def one_call(_i):
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        msg = resp.choices[0].message
        # Qwen3 is a reasoning model: include reasoning trace if present so
        # that differences anywhere in the generation are caught.
        reasoning = getattr(msg, "reasoning_content", None) or ""
        content = msg.content or ""
        return reasoning + "\x00" + content

    print(f"Sending {args.n} requests (concurrency={args.concurrency}) "
          f"to {args.base_url}")
    print(f"  model={args.model}  temperature={args.temperature}  "
          f"seed={args.seed}  max_tokens={args.max_tokens}")
    print(f"  prompt: {args.prompt!r}\n")

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            outputs = list(ex.map(one_call, range(args.n)))
    except Exception as e:
        print(f"ERROR talking to server: {e}", file=sys.stderr)
        sys.exit(1)

    digests = Counter(hashlib.sha256(o.encode()).hexdigest() for o in outputs)
    by_digest = {hashlib.sha256(o.encode()).hexdigest(): o for o in outputs}

    n_unique = len(digests)
    print(f"{'=' * 60}")
    print(f"Total responses : {len(outputs)}")
    print(f"Unique answers  : {n_unique}")
    verdict = "DETERMINISTIC ✓" if n_unique == 1 else "NON-DETERMINISTIC ✗"
    print(f"Verdict         : {verdict}")
    print(f"{'=' * 60}")

    print("\nDistribution (count x variant):")
    for i, (digest, count) in enumerate(digests.most_common(), 1):
        print(f"  variant {i}: {count:>3} occurrences  [{digest[:12]}]")

    if args.show_answers or n_unique > 1:
        print("\n--- unique answers ---")
        for i, (digest, _) in enumerate(digests.most_common(), 1):
            text = by_digest[digest].split("\x00", 1)[-1]  # content only
            print(f"\n[variant {i}]\n{text}")

    sys.exit(0 if n_unique == 1 else 2)


if __name__ == "__main__":
    main()
