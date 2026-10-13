"""Re-generate completions from a target model served by SGLang.

Adapted from SpecForge's scripts/regenerate_train_data.py with **id-keyed resume**: the
upstream `--resume` counts output lines and skips that many *input* lines from the top,
which assumes output order == input order. It doesn't — results are written by the
thread pool in completion order, so positional resume skips the wrong prompts. Here,
resume reads the ids already present in the output (and error) file and skips those, so
it is correct under concurrent out-of-order writes, safe across crashes, and safe to
extend: feed a larger input slice and only the new ids generate.

scripts/run_regen.sh starts the servers and runs this with the settings used for the
training data (thinking off, temperature 0.7, top-p 0.95, at most 3,072 new tokens).
"""

import argparse
import json
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True)
    model_group.add_argument(
        "--is-reasoning-model",
        action="store_true",
        help="Whether the model is a reasoning model",
    )
    model_group.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen3 thinking via chat_template_kwargs.enable_thinking=False",
    )

    sampling_params_group = parser.add_argument_group("sampling parameters")
    sampling_params_group.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sglang model server"
    )
    sampling_params_group.add_argument(
        "--top-p", type=float, default=None, help="Nucleus sampling top_p"
    )
    sampling_params_group.add_argument(
        "--top-k", type=int, default=None, help="Top-k sampling value sent via extra_body"
    )
    sampling_params_group.add_argument(
        "--repetition-penalty", type=float, default=None, help="Mapped to presence_penalty in the OpenAI API"
    )
    sampling_params_group.add_argument(
        "--max-tokens", type=int, default=4096, help="Maximum number of tokens (default: 4096)"
    )

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Requests in flight per server; total = concurrency * number of servers",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--input-file-path", type=str, required=True, help="Path to the input file")
    data_group.add_argument("--output-file-path", type=str, required=True, help="Path to the output file")
    data_group.add_argument(
        "--num-samples", type=int, default=None, help="Cap on new samples generated this run"
    )
    data_group.add_argument(
        "--resume",
        action="store_true",
        help="Skip ids already present in the output (and error) file; append rather than overwrite",
    )
    data_group.add_argument(
        "--retry-errors",
        action="store_true",
        help="On resume, re-attempt ids that previously errored (default: treat them as done)",
    )

    server_group = parser.add_argument_group("sglang server")
    server_group.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        help="Server address and port for sglang model server",
    )
    return parser.parse_args()


def compute_context_length(conversations: list[dict[str, Any]]) -> int:
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        length += len(text.split())
    return length


def build_query_kwargs(args, messages, max_tokens=None):
    effective_max_tokens = max_tokens if max_tokens is not None else args.max_tokens
    query_kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.no_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    return query_kwargs


_CLIENTS: dict[str, Any] = {}
_CLIENTS_LOCK = threading.Lock()


def sglang_client(server_address: str):
    # One client per server, shared by the worker threads: a client per request opens a
    # fresh connection pool each time and exhausts the process's file descriptors.
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(server_address)
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")
            _CLIENTS[server_address] = client
        return client


def call_sglang(args, server_address: str, data: dict[str, Any], max_tokens=None) -> dict[str, Any]:
    client = sglang_client(server_address)

    messages = data["conversations"]
    regenerated_messages = []

    if messages[0]["role"] == "assistant":
        data["status"] = "error"
        data["error"] = "Data starts with an assistant message"
        return data

    for message in messages:
        if message["role"] == "system":
            regenerated_messages.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            regenerated_messages.append(message)
            query_kwargs = build_query_kwargs(args, regenerated_messages, max_tokens)
            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data
            resp_msg = {"role": "assistant", "content": resp.choices[0].message.content}
            if args.is_reasoning_model:
                resp_msg["reasoning_content"] = resp.choices[0].message.reasoning_content
            regenerated_messages.append(resp_msg)
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data
    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


def load_done_ids(output_path, error_path, retry_errors):
    done = set()
    paths = [output_path] if retry_errors else [output_path, error_path]
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a half-written final line from a crash
                if "id" in rec:
                    done.add(str(rec["id"]))
    return done


def main():
    args = parse_arguments()

    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")
    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")

    error_file_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    done_ids = load_done_ids(args.output_file_path, error_file_path, args.retry_errors) if args.resume else set()
    file_mode = "a" if (args.resume and os.path.exists(args.output_file_path)) else "w"

    print("Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Max tokens: {args.max_tokens}  Temperature: {args.temperature}  top_p: {args.top_p}")
    print(f"  Concurrency: {args.concurrency}  Servers: {args.server_address}")
    print(f"  Input: {args.input_file_path}")
    print(f"  Output: {args.output_file_path} (mode {file_mode})")
    print(f"  Resume: {args.resume} ({len(done_ids)} ids already done, retry_errors={args.retry_errors})")
    print("-" * 50)

    total_lines = sum(1 for _ in open(args.input_file_path))

    valid_server_addresses = []
    for server_address in args.server_address:
        dummy = dict(conversations=[{"role": "user", "content": "Hello, how are you?"}])
        probe = call_sglang(args, server_address, dummy, max_tokens=1)
        if probe["status"] == "success":
            valid_server_addresses.append(server_address)
        else:
            print(f"Server {server_address} is not available: {probe['error']}")
    if not valid_server_addresses:
        raise ValueError("No server address is available")
    print(f"Using {len(valid_server_addresses)} servers: {valid_server_addresses}")
    print("-" * 50)

    context_token_sum = 0
    context_token_min = None
    context_token_max = 0
    success_samples = 0
    error_samples = 0
    skipped = 0

    def record_result(regen_data, output_handle, error_handle):
        nonlocal context_token_sum, context_token_min, context_token_max, success_samples, error_samples
        if regen_data["status"] == "error":
            error_handle.write(json.dumps(regen_data, ensure_ascii=False) + "\n")
            error_samples += 1
        else:
            ctx_len = compute_context_length(regen_data.get("conversations", []))
            context_token_sum += ctx_len
            context_token_min = ctx_len if context_token_min is None else min(context_token_min, ctx_len)
            context_token_max = max(context_token_max, ctx_len)
            output_handle.write(json.dumps(regen_data, ensure_ascii=False) + "\n")
            success_samples += 1

    with (
        open(args.input_file_path) as input_file,
        open(args.output_file_path, file_mode) as output_file_handle,
        open(error_file_path, file_mode) as error_file_handle,
    ):
        executor = ThreadPoolExecutor(max_workers=args.concurrency * len(valid_server_addresses))
        in_flight = {addr: set() for addr in valid_server_addresses}
        pbar = tqdm(total=total_lines, desc="Processing")

        def reap(done):
            for futures in in_flight.values():
                for req_future in futures & done:
                    record_result(req_future.result(), output_file_handle, error_file_handle)
                futures -= done

        for line in input_file:
            pbar.update(1)
            if args.num_samples is not None and success_samples + error_samples >= args.num_samples:
                break

            data = json.loads(line.strip())
            if str(data.get("id")) in done_ids:
                skipped += 1
                continue

            # Each prompt goes to the server with the most free slots, so a slow server never
            # holds back a fast one.
            server_address = min(valid_server_addresses, key=lambda a: len(in_flight[a]))
            while len(in_flight[server_address]) >= args.concurrency:
                done, _ = wait(set().union(*in_flight.values()), return_when=FIRST_COMPLETED)
                reap(done)
                server_address = min(valid_server_addresses, key=lambda a: len(in_flight[a]))

            in_flight[server_address].add(executor.submit(call_sglang, args, server_address, data))

        reap(wait(set().union(*in_flight.values()))[0])

    print(f"\nDone. Skipped {skipped} already-generated ids.")
    print(f"Newly generated: {success_samples} success, {error_samples} error.")
    if success_samples > 0:
        print(
            f"Context length (untokenized words) over new samples: "
            f"min {context_token_min}, max {context_token_max}, avg {context_token_sum / success_samples:.1f}"
        )
    if error_samples:
        raise SystemExit(f"{error_samples} prompts failed (see {error_file_path}); rerun with --resume --retry-errors")


if __name__ == "__main__":
    main()
