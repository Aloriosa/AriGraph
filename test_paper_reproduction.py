#!/usr/bin/env python3

import re
import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from time import time, sleep
import datetime

import numpy as np
from graphs.contriever_graph import ContrieverGraph
from utils.retriever_search_drafts import graph_retr_search
from utils.utils import Logger, clear_triplet

from openai import OpenAI
from dotenv import load_dotenv # you'll need to pip install python-dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
import prompts.cookbook_extraction_prompt as prmt
from ontology.cookbook_ontology import validate_triplet
import utils_reproduction as reprutil
from topology_pipeline.cards import build_memory_cards, build_paper_cards, cards_to_dict
from topology_pipeline.graph_materialization import materialize_typed_graph
from topology_pipeline.prompt_packing import PromptBudget, build_generation_prompt
from topology_pipeline.retrieval import RetrievalConfig, retrieve_memory_cards
from topology_pipeline.symbol_assets import build_symbol_assets_from_triplet_links


def str2bool(v):
    """Explicit two-way bool for CLI flags: --foo true / --foo false."""
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def _emb_sidecar_path(json_path: str) -> str:
    """Return the on-disk path for a JSON file's triplet-embedding cache sidecar."""
    if json_path.endswith(".json"):
        return json_path[:-5] + "_emb.npz"
    return json_path + "_emb.npz"


def _emb_map_to_array(emb_map):
    """Stack a {key: vector} map into (keys_array, embs_array); (None, None) if shapes differ."""
    keys = list(emb_map.keys())
    arrs = []
    for k in keys:
        v = emb_map[k]
        if hasattr(v, "cpu"):
            v = v.cpu().detach().numpy()
        arrs.append(np.asarray(v))
    try:
        embs = np.stack(arrs)
    except ValueError:
        return None, None
    return np.array(keys, dtype=object), embs


def _load_graph_embeddings(json_path: str, log) -> dict:
    """Load triplet/item/observation embedding caches from one sidecar.

    Returns {'triplets':{}, 'items':{}, 'obs':{}}. Back-compatible with the old
    triplets-only format (keys/embs). Missing or corrupt sidecar -> empty maps;
    we never raise — a miss just means re-embedding.
    """
    out = {"triplets": {}, "items": {}, "obs": {}}
    side = _emb_sidecar_path(json_path)
    if not os.path.isfile(side):
        return out
    try:
        npz = np.load(side, allow_pickle=True)

        def grp(kk, ek):
            if kk in npz.files and ek in npz.files and len(npz[kk]) == len(npz[ek]):
                return {str(k): npz[ek][i] for i, k in enumerate(npz[kk])}
            return {}

        out["triplets"] = grp("trip_keys", "trip_embs") or grp("keys", "embs")  # old format
        out["items"] = grp("item_keys", "item_embs")
        out["obs"] = grp("obs_keys", "obs_embs")
        log(f"T4: restored embeddings from {side} "
            f"(triplets={len(out['triplets'])}, items={len(out['items'])}, obs={len(out['obs'])})")
    except Exception as e:
        log(f"T4: failed to read embedding cache at {side}: {e}; will re-embed.")
    return out


def save_graph_embeddings(json_path: str, graph, log) -> None:
    """Atomically persist triplet, entity-label, and observation-key embeddings.

    obs embeddings are the second element of each obs_episodic value ([text, embedding]).
    """
    trip = getattr(graph, "triplets_emb", {})
    items = getattr(graph, "items_emb", {})
    obs = {k: v[1] for k, v in getattr(graph, "obs_episodic", {}).items()
           if isinstance(v, (list, tuple)) and len(v) > 1}
    payload = {}
    for name, m in (("trip", trip), ("item", items), ("obs", obs)):
        if not m:
            continue
        keys, embs = _emb_map_to_array(m)
        if keys is None:
            log(f"T4: heterogeneous {name} embedding shapes, skipping {name} group.")
            continue
        payload[f"{name}_keys"] = keys
        payload[f"{name}_embs"] = embs
    if not payload:
        return
    side = _emb_sidecar_path(json_path)
    tmp = side + ".tmp"
    # File-object so numpy doesn't auto-append ".npz" to our ".tmp" suffix.
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, side)
    log(f"T4: saved embeddings to {side} (triplets={len(trip)}, items={len(items)}, obs={len(obs)})")


class ReproductionGraph(ContrieverGraph):
    """Extended ContrieverGraph that uses reproduction-focused prompts."""

    def __init__(self, model, system_prompt, api_key, log, base_url='', device="cpu", type='paper'): # or type 'mem'
        super().__init__(model, system_prompt, api_key, base_url, device)
        if type == 'mem':
            self.graph_builder_instruction = (
                "Extract only REUSABLE patterns for a cumulative cookbook. "
                "Skip paper-specific details. This graph helps implement OTHER papers in the area.\n\n"
            )
            self.reproduction_prompt = prmt.prompt_cookbook_reusable_extraction
        elif type == 'paper':
            self.graph_builder_instruction = (
                "Extract all the important information required to reproduce the method described in the paper with all the quantitative and qualitative results.\n\n"
            )
            self.reproduction_prompt = prmt.prompt_cookbook_extraction_wo_code
        else:
            raise ValueError(f"Invalid type: {type}")
        self.hypernode_store = {}
        self._log = log
        log(f'\nPrompt: {self.reproduction_prompt[:200]}...')
        self.completion_tokens  = 0
        self.prompt_tokens  = 0
        self.triplet2code = {}
        self.emb_cache = True  # write/read triplet-embedding sidecars; toggled via --emb-cache

    
    def _add_triplets_with_cached_embeddings(self, triplets, cached_emb, cached_items=None):
        """Like `add_triplets` but reuses pre-computed triplet AND entity-label embeddings.

        cached_items: {entity_label: embedding}; entities absent from it are re-embedded.
        """
        cached_items = cached_items or {}
        re_embed_trip = 0
        re_embed_item = 0
        for triplet in triplets:
            if not isinstance(triplet, (list, tuple)) or len(triplet) < 3:
                continue
            t = clear_triplet(triplet)
            if t[2].get("label") == "free":
                continue
            if t in self.triplets:
                continue
            self.triplets.append(t)
            key = self.str(t)
            if key in cached_emb:
                self.triplets_emb[key] = cached_emb[key]
            else:
                self.triplets_emb[key] = self.get_embedding_local(key)
                re_embed_trip += 1
            for ent in (t[0], t[1]):
                if ent in self.items_emb:
                    continue
                if ent in cached_items:
                    self.items_emb[ent] = cached_items[ent]
                else:
                    self.items_emb[ent] = self.get_embedding_local(ent)
                    re_embed_item += 1
        if re_embed_trip or re_embed_item:
            self._log(f"T4: re-embedded {re_embed_trip} triplets, {re_embed_item} items not in cache.")
    
    def add_triplets(self, triplets, ontology_validation=False):
        """Override to enforce ontology validation before adding."""
        validated = []
        if ontology_validation:
            for t in triplets:
                if validate_triplet(t):
                    validated.append(t)
                else:
                    self._log(f"Skipping invalid triplet (ontology): {t}")
        else:
            validated = triplets
        if validated:
            super().add_triplets(validated)
        
    
    def update_without_retrieve(self, observation, prev_subgraph, log, source_type="paper"):
        """Override to use reproduction-focused prompt."""
        example = [re.sub(r"Step \d+: ", "", t) for t in prev_subgraph]
        example_str = "; ".join(example) if example else ""
        
        if source_type == "code":
            observation = f"[CODE IMPLEMENTATION]\n{observation}"
        else:
            observation = f"[PAPER TEXT]\n{observation}"
        
        prompt = self.graph_builder_instruction + self.reproduction_prompt.format(
            observation=observation, 
            example=example_str
        )
        #self._log(f'\nPrompt: {prompt}...')
        
        response, tokens = self.generate(prompt, t=0.001)
        self.completion_tokens += tokens["completion_tokens"]
        self.prompt_tokens += tokens["prompt_tokens"]
        #log('response generated')
        # Process triplets (reuse parent class logic)
        from utils.utils import process_triplets
        new_triplets_raw = process_triplets(response)
        new_triplets = self.exclude(new_triplets_raw)
        new_triplets_str = self.convert(new_triplets_raw)
        
        self.add_triplets(new_triplets_raw)
        #log('triplets processed')
        
        n_added = len(new_triplets)
        n_parsed = len(new_triplets_raw)
        if n_added < n_parsed:
            log(f"Parsed {n_parsed} triplets from {source_type}, {n_added} new (added to graph), {n_parsed - n_added} duplicates or invalid")
        else:
            log(f"Parsed {n_parsed} triplets from {source_type}, {n_added} new")
        
        obs_embedding = self.retriever.embed(observation)
        obs_value = [new_triplets_str, obs_embedding]
        self.obs_episodic[observation] = obs_value
        
        return new_triplets_raw, obs_value

    def load_triplets_from_json(self, path, clear_first=True):
        path = os.path.abspath(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._log(f"Loaded data from {path} with keys: {data.keys()}")
        triplets = data.get("triplets", [])
        if not triplets:
            self._log(f"No triplets found in {path}")
        else:
            if clear_first:
                self.clear()
            
            # T4: if a sidecar embedding cache exists, hydrate directly (skip re-embed).
            cache = _load_graph_embeddings(path, self._log)
            if cache["triplets"]:
                self._add_triplets_with_cached_embeddings(triplets, cache["triplets"], cache["items"])
            else:
                self.add_triplets(triplets)
            n = len(self.triplets)
            obs_episodic = data.get("obs_episodic", [])
            if not obs_episodic:
                self._log(f"No triplets found in {path}")
            else:
                # Reuse cached observation-key embeddings; embed only the misses.
                cobs = cache["obs"]
                self.obs_episodic = {
                    k: [v, cobs[k] if k in cobs else self.retriever.embed(k)]
                    for k, v in obs_episodic.items()
                }
            # Backfill / upgrade the sidecar when anything was missing (old format or first load).
            if getattr(self, "emb_cache", True) and (
                not cache["triplets"] or not cache["items"] or not cache["obs"]
            ):
                save_graph_embeddings(path, self, self._log)
            if path.endswith("mem_graph_data_with_code.json"):
                triplet2code = data.get("triplet2code", {})
                if not triplet2code:
                    self._log(f"No code- linking found in {path}")
                else:
                    self.triplet2code = triplet2code

            self._log(f"Loaded {n} triplets from {path}")
        return n

    def retrieve(self, queries, topk=3):
        triplets = self.triplets_to_str(self.triplets)
        
        #associated_subgraph = set()
        #retrieve for dict of items
        #print(f"retrieve  items: {queries[:10]}")

        results = graph_retr_search(
            queries, triplets, self.retriever, 
            topk=topk,
            post_retrieve_threshold=0.75, 
            verbose=2
        )
        #associated_subgraph.update(results)

        #associated_subgraph = [element for element in associated_subgraph]
        assert len(results) == len(queries), f"len(results) != len(queries): {len(results)} != {len(queries)}"
        return results # associated_subgraph
    


def run_linking_pass_hypernodes(graph, code_files, log,
    output_base=None, paper_path=None, repo_dir=None,
    observation_keys_filter=None, only_new_patterns=False):
    """
    observation_keys_filter: if set, only run linking for these observation keys (e.g. new code chunks).
    only_new_patterns: if True, never add or overwrite triplet2code[pattern] when pattern already has a link
    (keeps the theory graph free of duplicate / upgraded code links for the same triplet line).
    """

    symbol_index_full = reprutil.build_symbol_index_text(code_files)
    symbol_records = reprutil.extract_symbol_records(code_files)
    if output_base:
        reprutil._write_json_to_path(
            output_base,
            "symbol_records.json",
            {"symbols": symbol_records, "count": len(symbol_records)},
        )
    if len(symbol_index_full) == 0:
        log('no code files')
        return graph
    max_seq_len = int(os.getenv("MODEL_MAX_SEQ_LEN"))
    obs_idx = 0
    episodic_items = list(graph.obs_episodic.items())
    total_obs = len(episodic_items)
    for observation, (new_triplets_str, _) in episodic_items:
        if observation_keys_filter is not None and observation not in observation_keys_filter:
            continue
        if len(new_triplets_str) == 0:
            log('no triplets in obs')
            continue
        obs_idx += 1
        log(f"Processing observation {obs_idx} (episodic entries in graph: {total_obs})")
        numbered_triplets = reprutil.format_numbered_triplet_lines(new_triplets_str)
        sym_start = 0
        seen_pattern_location_pairs = set()
        while sym_start < len(symbol_index_full):
            
            # 1 token ~ 4 chars
            template = prmt.prompt_code_linking_symbol_index.format(
                observation=observation,
                numbered_triplets=numbered_triplets,
                symbol_index='',
            )
            chars_left = int(3.8 * (max_seq_len - reprutil.estimate_tokens(
                template,
                model=os.getenv("OPENAI_MODEL"))))
            #log(f"Chars left: {chars_left}")
            end = sym_start + chars_left

            sym_chunk = symbol_index_full[sym_start:end]
            prompt = prmt.prompt_code_linking_symbol_index.format(
                observation=observation,
                numbered_triplets=numbered_triplets,
                symbol_index=sym_chunk,
            )
            # The 3.8 chars/token guess underestimates code-dense text (~2 chars/token),
            # so the char-sliced chunk can still exceed the model context. Shrink against the
            # real token count (budget = max_seq_len, the served context) before sending;
            # rescale by the measured ratio with a safety margin, never grow.
            prompt_tokens = reprutil.estimate_tokens(prompt, model=os.getenv('OPENAI_MODEL'))
            while prompt_tokens > max_seq_len and len(sym_chunk) > 0:
                end = sym_start + int(len(sym_chunk) * (max_seq_len / prompt_tokens) * 0.95)
                sym_chunk = symbol_index_full[sym_start:end]
                prompt = prmt.prompt_code_linking_symbol_index.format(
                    observation=observation,
                    numbered_triplets=numbered_triplets,
                    symbol_index=sym_chunk,
                )
                prompt_tokens = reprutil.estimate_tokens(prompt, model=os.getenv('OPENAI_MODEL'))
            log(f"Prompt tokens: {prompt_tokens}")

            
            log(
                f"  Chars [{sym_start}:{end}] of {len(symbol_index_full)}"
            )

            
            response, _ = graph.generate(prompt, jsn=True, t=0.001)
            if response:
                links = reprutil.parse_linking_json_response(response)
                for link in links:
                    if not isinstance(link, dict):
                        continue
                    tid = link.get("triplet_id")
                    code_loc = link.get("code_location")
                    conf = link.get("confidence")
                    if tid is None or not code_loc:
                        continue
                    tid = int(tid)
                    lines = list(new_triplets_str)
                    if tid < 1 or tid > len(lines):
                        log(f"  Warning: triplet_id {tid} out of range (1..{len(lines)})")
                        continue
                    
                    try:
                        conf_f = float(conf) if conf is not None else 0.0
                    except (TypeError, ValueError):
                        conf_f = 0.0
                    
                    pattern = lines[tid - 1]
                    loc_key = str(code_loc).strip()
                    pl_key = (pattern, loc_key)
                    if pl_key in seen_pattern_location_pairs:
                        continue
                    if only_new_patterns and pattern in graph.triplet2code:
                        continue
                    if pattern in graph.triplet2code and graph.triplet2code[pattern]['confidence'] > conf_f:
                        continue
                    
                    snip = reprutil.snippet_from_file_line(repo_dir, loc_key)
                    code_val = snip.get("code", '')
                    err = snip.get("error")
                    if err:
                        log(f"  Could not resolve {code_loc!r}: {err}")
                    if not code_val:
                        continue
                    graph.triplet2code[pattern] = {
                        "code": code_val,
                        "code_location": loc_key,
                        "confidence": conf_f,
                        "line_start": snip.get("line_start", 0),
                        "line_end": snip.get("line_end", 0),
                        "path": snip.get("path", ""),
                        "documentation": "",
                        "imports": [],
                        "paper": paper_path,
                    }
                    seen_pattern_location_pairs.add(pl_key)
                    log(f"  Obs {obs_idx}, sym_start {sym_start}: Linked triplet [{tid}] -> {code_loc} (confidence={round(conf_f, 3)})")

                if output_base and paper_path:
                    reprutil._write_json_to_path(
                        output_base,
                        "mem_graph_data_with_code.json",
                        {
                            "paper": os.path.basename(paper_path),
                            "triplets": [[t[0], t[1], t[2]] for t in graph.triplets],
                            "triplet2code": graph.triplet2code,
                            "stats": {
                                "total_triplets": len(graph.triplets),
                                "prompt_tokens": graph.prompt_tokens,
                                "completion_tokens": graph.completion_tokens,
                            },
                        },
                    )

            # Advance only after this chunk is processed; full symbol text is covered with no truncation.
            sym_start = end
            
    log(f" Unique hypernodes added: {len(graph.triplet2code)}")
    return graph


def _parse_code_generator_output(response_text, repo_root):
    """
    Parse LLM output in format:
      FILE: path/to/file.ext
      ```language
      <content>
      ```

    Creates the repo structure under repo_root, preserves file names, extensions, and content.
    Returns (files_created: list of Path, primary_python: Path or None).
    """
    files_created = []
    primary_python = None

    # Pattern 1: FILE: path (allow any prefix before FILE:, e.g. ##, **, -, 1., etc.)
    file_decl_re = re.compile(r"^\s*.*?FILE:\s*(.+?)\s*\*{0,2}\s*$", re.MULTILINE | re.IGNORECASE)
    # Pattern 2: **/path/to/file** (no FILE: - some LLMs use this format)
    #file_decl_alt_re = re.compile(r"^\s*\*{1,2}\s*(/[a-zA-Z0-9_/.-]+)\s*\*{0,2}\s*$", re.MULTILINE)
    # Fenced code block: ```lang\n...``` (content can span lines)
    code_block_re = re.compile(r"^```(\w*)\s*\n([\s\S]*?)```", re.MULTILINE)
    fence_re = re.compile(r"^```", re.MULTILINE)

    # Find FILE: declarations that are NOT inside fenced code blocks (e.g. markdown
    # content often contains code blocks with "FILE: path" as documentation)
    def _is_inside_code_block(pos):
        # Build (start, end) for each code block; fences can nest (e.g. ```markdown with ```bash inside)
        fences = list(fence_re.finditer(response_text))
        blocks = []
        stack = []
        for m in fences:
            p = m.start()
            rest = response_text[p + 3 :]
            is_opening = rest and (rest[0].isalnum() or rest[0] == "_")
            if is_opening:
                stack.append(p)
            else:
                if stack:
                    start = stack.pop()
                    blocks.append((start, p + 3))
        return any(start <= pos < end for start, end in blocks)

    def _unwrap_markdown_segment(segment):
        """If segment is wrapped in ```markdown ... ```, extract inner content.
        The segment already ends at the next FILE: declaration, so we use the LAST ```
        in the segment as the closing fence - this correctly handles nested code blocks
        (e.g. ```bash, plain ```) inside the markdown."""
        segment = segment.strip()
        if not re.match(r"^```(?:markdown|md)\s*\n", segment, re.IGNORECASE):
            return segment
        start = re.match(r"^```\w*\s*\n", segment, re.IGNORECASE).end()
        # Find the last ``` at start of line - that closes the outer markdown block
        last_fence = -1
        pos = 0
        while True:
            idx = segment.find("```", pos)
            if idx == -1:
                break
            if idx == 0 or segment[idx - 1] == "\n":
                last_fence = idx
            pos = idx + 3
        if last_fence >= 0:
            return segment[start:last_fence].rstrip()
        return segment

    # Collect matches from both patterns, filter out those inside code blocks
    file_matches = [m for m in file_decl_re.finditer(response_text) if not _is_inside_code_block(m.start())]
    #alt_matches = [m for m in file_decl_alt_re.finditer(response_text) if not _is_inside_code_block(m.start())]
    # Merge and sort by position; for alt pattern, use group(1) as path (need to extract path from match)
    #for m in alt_matches:
    #    if not any(fm.start() == m.start() for fm in file_matches):  # avoid duplicate
    #        file_matches.append(m)
    file_matches.sort(key=lambda m: m.start())

    if not file_matches and ("FILE" in response_text.upper() or "**/" in response_text):
        logging.warning(
            "Parse: 'FILE' found in response but no FILE: declarations matched regex; "
            "falling back to single code block or raw text"
        )

    for i, file_match in enumerate(file_matches):
        raw_path = file_match.group(1).strip()
        # Normalize: forward slashes, strip leading slashes
        raw_path = raw_path.replace("\\", "/").lstrip("/")
        # repo_root already IS the submission dir, but the model writes paths against the
        # eval container's /home/submission/ root. Strip that prefix so files land directly
        # under submission/ instead of double-nesting into submission/home/submission/.
        if raw_path.startswith("home/submission/"):
            raw_path = raw_path[len("home/submission/"):]
        if not raw_path:
            continue
        # Sanitize: avoid path traversal
        parts = raw_path.split("/")
        safe_parts = [p for p in parts if p and p != ".."]
        rel_path = "/".join(safe_parts) if safe_parts else raw_path
        if not rel_path:
            continue

        start = file_match.end()
        next_file_start = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(response_text)
        segment = response_text[start:next_file_start]

        if rel_path.lower().endswith(".md"):
            content = _unwrap_markdown_segment(segment)
        else:
            code_match = code_block_re.search(segment)
            if not code_match:
                continue
            content = code_match.group(2).rstrip()
        if content.endswith("\n"):
            content = content[:-1]  # Keep trailing newline behavior consistent

        out_path = Path(repo_root) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        #try:
        out_path.write_text(content, encoding="utf-8")
        #except Exception as e:
        #    out_path.write_text(content, encoding="utf-8", errors="replace")
        #    raise e
        files_created.append(out_path)
        if primary_python is None and out_path.suffix.lower() in (".py", ".pyw"):
            primary_python = out_path

    # Fallback: if no FILE: blocks found, try single fenced code block (legacy)
    if not files_created:
        for match in code_block_re.finditer(response_text):
            content = match.group(2).rstrip()
            lang = (match.group(1) or "").lower()
            ext = ".py" if "python" in lang or lang == "py" else ".sh" if "bash" in lang or "sh" in lang else ".txt"
            out_path = Path(repo_root) / f"generated_reproduction{ext}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                out_path.write_text(content, encoding="utf-8")
            except Exception:
                out_path.write_text(content, encoding="utf-8", errors="replace")
            files_created.append(out_path)
            primary_python = out_path if ext == ".py" else primary_python
            break
        if not files_created:
            # Last resort: raw text as single file
            out_path = Path(repo_root) / "generated_reproduction.py"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(response_text.strip(), encoding="utf-8", errors="replace")
            files_created.append(out_path)
            primary_python = out_path

    return files_created, primary_python


def _format_triplets_for_prompt(triplets, max_triplets=None):
    """Format triplets as bullet list for LLM prompt."""
    lines = []
    for row in triplets[:max_triplets if max_triplets is not None else len(triplets)]:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            s, o, r = row[0], row[1], row[2]
            rel = r.get("label", r) if isinstance(r, dict) else str(r)
            lines.append(f"- {s}, {rel}, {o}")
        elif isinstance(row, tuple) and len(row) == 3:
            s, o, r = row
            rel = r.get("label", r) if isinstance(r, dict) else str(r)
            lines.append(f"- {s}, {rel}, {o}")
    # remove duplicates
    lines = list(set(lines))
    return "\n".join(lines) if lines else "(none)"


def _format_expert_graph_for_agent(triplets, hypernode_store, max_triplets=None):
    """Format expert triplets + hypernodes for the coding agent (same logic as _format_cookbook_for_agent)."""
    parts = []
    pattern_lines = []
    impl_triplets = []
    for row in triplets[:max_triplets if max_triplets is not None else len(triplets)]:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            s, o, r = row[0], row[1], row[2]
        else:
            continue
        rel = r.get("label", r) if isinstance(r, dict) else str(r)
        line = f"{s}, {rel}, {o}"
        if rel == "implemented_in":
            impl_triplets.append((s, o))
        else:
            pattern_lines.append(line)

    seen = set()
    unique = []
    for line in pattern_lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    parts.append("## Patterns and config")
    parts.append("\n".join(f"- {line}" for line in unique[:max_triplets if max_triplets is not None else len(unique)]))

    if hypernode_store and impl_triplets:
        parts.append("\n## Pattern -> Code (hypernodes)")
        for pattern, hid in impl_triplets:
            if hid not in hypernode_store:
                continue
            hn = hypernode_store[hid]
            code = hn.get("code", "")
            doc = hn.get("documentation", "")
            imports = hn.get("imports", [])
            if not isinstance(imports, list):
                imports = [imports] if imports else []
            parts.append(f"\n### {pattern} (implemented_in)")
            parts.append(f"Documentation: {doc}")
            parts.append(f"Imports: {', '.join(imports)}")
            if code:
                parts.append(f"Code:\n```\n{code}\n```")
    return "\n".join(parts)


def generate_code_from_graph(graph, paper_path, paper_name, log, output_dir=None, max_triplets=None,
                             paper_summary=None, hypernode_store=None, graph_base_dir=None):
    """
    Generate implementation code using paper graph + expert graph.
    Loads paper_graph_data.json and graph_data.json from graph_base_dir (or output_dir if None).
    Writes submission, generated_code_response to output_dir.
    """
    output_dir = graph_base_dir if output_dir is None else output_dir
    graph_data_path = os.path.join(graph_base_dir, "graph_data.json")
    paper_graph_path = os.path.join(graph_base_dir, "paper_graph_data.json")
    
    paper_graph_block = ""
    if os.path.exists(paper_graph_path):
        with open(paper_graph_path, "r", encoding="utf-8") as f:
            paper_data = json.load(f)
        paper_triplets = paper_data.get("triplets", [])
        paper_graph_block = _format_triplets_for_prompt(paper_triplets, max_triplets)
    expert_graph_block = False
    if os.path.exists(graph_data_path):
        with open(graph_data_path, "r", encoding="utf-8") as f:
            expert_data = json.load(f)
        expert_triplets = expert_data.get("triplets", [])
        expert_hypernodes = expert_data.get("hypernode_store", {})
        expert_graph_block = _format_expert_graph_for_agent(
            expert_triplets, expert_hypernodes, max_triplets
        )

    if not paper_summary and paper_path:
        
        with open(paper_path, "r", encoding="utf-8") as f:
            full = f.read()
        paper_summary = full[:2000] + ("..." if len(full) > 2000 else "")
        
    if not paper_summary:
        paper_summary = "None"

    
    if paper_graph_block and expert_graph_block:
        prompt = prmt.prompt_coding_agent_paper_mem_graph.format(
            BENCHMARK_RULES=prmt.BENCHMARK_EVALUATION_RULES,
            TOY_EXAMPLE=prmt.REPRODUCTION_SCRIPT_TOY_EXAMPLE,
            paper_summary=paper_summary,
            paper_graph=paper_graph_block,
            expert_graph=expert_graph_block,
        )
    elif paper_graph_block:
        prompt = prmt.prompt_coding_agent_paper_graph.format(
            BENCHMARK_RULES=prmt.BENCHMARK_EVALUATION_RULES,
            TOY_EXAMPLE=prmt.REPRODUCTION_SCRIPT_TOY_EXAMPLE,
            paper_summary=paper_summary,
            paper_graph=paper_graph_block,
        )
    log(f"Prompt length: {len(prompt)} chars: benchmark rules {len(prmt.BENCHMARK_EVALUATION_RULES)} chars, toy example {len(prmt.REPRODUCTION_SCRIPT_TOY_EXAMPLE)} chars, paper summary {len(paper_summary)} chars, paper graph {len(paper_graph_block)} chars")
    response, tokens = graph.generate(prompt)
    completion_tokens = tokens["completion_tokens"]
    prompt_tokens = tokens["prompt_tokens"]
    log(f"Code generation response: {len(response)} chars")
    log(f"Code generation tokens: {completion_tokens} completion, {prompt_tokens} prompt")

    os.makedirs(output_dir, exist_ok=True)
    raw_output_path = os.path.join(output_dir, "generated_code_response.txt")
    repo_root = os.path.join(output_dir, paper_name, "submission")
    os.makedirs(repo_root, exist_ok=True)

    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(response)

    files_created, primary_python = _parse_code_generator_output(response, repo_root)
    code_path = str(primary_python) if primary_python else (os.path.join(repo_root, "generated_reproduction.py") if files_created else None)

    for p in files_created:
        rel = Path(p).relative_to(Path(repo_root))
        log(f"  Created: {rel}")
    log(f"Generated repo at {repo_root} ({len(files_created)} files)")
    log(f"Raw model response saved to: {raw_output_path}")

    return {
        "code_path": code_path,
        "repo_root": repo_root,
        "files_created": [str(p) for p in files_created],
        "raw_response_path": raw_output_path,
        "paper_triplets": len(paper_graph_block),
        #"expert_triplets": len(expert_graph_block),
        "code_chars": sum(p.stat().st_size for p in files_created if p.exists()),
    }


def generate_code_from_graph_with_retrieval(mem_graph, paper_graph, log,
output_dir=None, max_triplets=None, paper_summary=None, 
graph_base_dir=None, retrieval_topk=8, retrieval_max_depth=2, retrieval_threshold=0.7, 
max_paper_queries=25, code_snippet_len=5000):
    """
    Generate implementation code using paper graph + RETRIEVED relevant snippets from expert graph.
    First uses paper graph to retrieve relevant code snippets from expert knowledge graph,
    then concatenates paper graph + retrieval results into the coding agent input.
    Loads from graph_base_dir (or output_dir if None); writes to output_dir.
    """
    
    agent = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    output_dir = graph_base_dir if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)

    log("Starting topology-aware retrieval")

    paper_typed = materialize_typed_graph(
        graph_kind="paper",
        raw_triplets=paper_graph.triplets,
        obs_episodic=paper_graph.obs_episodic,
    )
    mem_typed = materialize_typed_graph(
        graph_kind="memory",
        raw_triplets=mem_graph.triplets,
        obs_episodic=mem_graph.obs_episodic,
    )
    reprutil._write_json_to_path(output_dir, "paper_graph_typed.json", paper_typed.to_dict())
    reprutil._write_json_to_path(output_dir, "mem_graph_typed.json", mem_typed.to_dict())

    symbol_assets, triplet_to_asset = build_symbol_assets_from_triplet_links(mem_graph.triplet2code)
    paper_cards = build_paper_cards(paper_typed)
    memory_cards = build_memory_cards(mem_typed, symbol_assets, triplet_to_asset)
    reprutil._write_json_to_path(output_dir, "paper_cards.json", {"cards": cards_to_dict(paper_cards)})
    reprutil._write_json_to_path(output_dir, "mem_cards.json", {"cards": cards_to_dict(memory_cards)})
    reprutil._write_json_to_path(
        output_dir,
        "symbol_assets.json",
        {"assets": [asset.to_dict() for asset in symbol_assets], "triplet_to_asset": triplet_to_asset},
    )


    # Use the Contriever encoder we already built for the graph as the dense retriever.
    # Falls back to lexical Jaccard automatically if embedding fails.
    def _mem_embedder(texts):
        return mem_graph.retriever.embed(list(texts))

    selected = retrieve_memory_cards(
        paper_cards,
        memory_cards,
        RetrievalConfig(
            per_paper_card=max(1, min(4, retrieval_topk)),
            total_budget=max(8, max_paper_queries),
            min_score=max(0.01, min(0.3, retrieval_threshold / 3.0)),
            embedder=_mem_embedder,
        ),
    )
    reprutil._write_json_to_path(
        output_dir,
        "retrieval_selection.json",
        {"selection": [{"paper_card_id": p, "memory_card_id": m, "score": s} for p, m, s in selected]},
    )
    # Backward-compatible artifact used by existing analysis scripts.
    expert_data = {}
    paper_card_by_id = {card.id: card for card in paper_cards}
    memory_card_by_id = {card.id: card for card in memory_cards}
    for p_id, m_id, score in selected:
        p_card = paper_card_by_id.get(p_id)
        m_card = memory_card_by_id.get(m_id)
        if not p_card or not m_card:
            continue
        key = p_card.id
        expert_data.setdefault(key, {"triplets": p_card.generation_summary, "expert_knowledge": []})
        expert_data[key]["expert_knowledge"].append(
            {
                "Mem triplet": m_card.generation_summary,
                "Code snippet": "\n".join(m_card.linked_asset_ids),
                "score": score,
            }
        )
    reprutil._write_json_to_path(output_dir, "expert_data.json", expert_data)

    budget = PromptBudget(
        max_prompt_tokens=int(os.getenv("MODEL_MAX_SEQ_LEN", "262000")),
        reserved_output_tokens=int(os.getenv("MODEL_OUTPUT_TOKEN_RESERVE", "120000")),
        fixed_block_tokens=12000,
        paper_cards_tokens=28000,
        memory_cards_tokens=36000,
        code_assets_tokens=22000,
    )
    packed = build_generation_prompt(
        benchmark_rules=prmt.BENCHMARK_EVALUATION_RULES,
        toy_example=prmt.REPRODUCTION_SCRIPT_TOY_EXAMPLE,
        paper_cards=paper_cards,
        memory_cards=memory_cards,
        selected_map=selected,
        symbol_assets=symbol_assets,
        budget=budget,
    )
    prompt = prmt.prompt_coding_agent_paper_mem_graph.format(
        BENCHMARK_RULES=prmt.BENCHMARK_EVALUATION_RULES,
        TOY_EXAMPLE=prmt.REPRODUCTION_SCRIPT_TOY_EXAMPLE,
        expert_knowledge=(
            "## Paper cards\n"
            + packed["paper_block"]
            + "\n\n## Memory cards\n"
            + packed["memory_block"]
            + "\n\n## Linked code assets\n"
            + packed["code_asset_block"]
        ),
    )
    template_seq_len = reprutil.estimate_tokens(prompt, model=os.getenv("OPENAI_MODEL"))
    max_seq_len = int(os.getenv("MODEL_MAX_SEQ_LEN", "262000"))
    reserved = int(os.getenv("MODEL_OUTPUT_TOKEN_RESERVE", "120000"))
    max_prompt_allowed = max(8_000, max_seq_len - reserved)
    if template_seq_len > max_prompt_allowed:
        log(f"Prompt length {template_seq_len} exceeds allowed prompt budget {max_prompt_allowed}, truncating...")
        ratio = max_prompt_allowed / max(1, template_seq_len)
        keep_chars = int(len(prompt) * ratio * 0.98)
        prompt = prompt[:keep_chars]

    log(f"Prompt length: {reprutil.estimate_tokens(prompt,model=os.getenv('OPENAI_MODEL'))} TOKENS, generating code...")
    chat_completion = agent.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": 'You are a helpful assistant for research reproduction',
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=os.getenv("OPENAI_MODEL"),
                temperature=0.7,
                
            )
    response = chat_completion.choices[0].message.content
    # print(f"Response: {chat_completion}")
    prompt_tokens = chat_completion.usage.prompt_tokens
    completion_tokens = chat_completion.usage.completion_tokens

    log(f"Code generation response: {len(response)} chars")
    log(f"Code generation tokens: {completion_tokens} completion, {prompt_tokens} prompt")

    raw_output_path = os.path.join(output_dir, "generated_code_response.txt")
    repo_root = os.path.join(output_dir, "submission")
    os.makedirs(repo_root, exist_ok=True)

    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(response)

    files_created, primary_python = _parse_code_generator_output(response, repo_root)
    code_path = str(primary_python) if primary_python else (os.path.join(repo_root, "generated_reproduction.py") if files_created else None)

    for p in files_created:
        try:
            rel = Path(p).relative_to(Path(repo_root))
        except ValueError:
            rel = p
        log(f"  Created: {rel}")
    log(f"Generated repo at {repo_root} ({len(files_created)} files)")
    log(f"Raw model response saved to: {raw_output_path}")

    return {
        "code_path": code_path,
        "repo_root": repo_root,
        "files_created": [str(p) for p in files_created],
        "raw_response_path": raw_output_path,
    }


def run_reproduction_test(log, args, paper_path, device="cpu", log_path="",
                        generate_code=True, use_repo=True,
                        repo_dir=None, code_gen_variant="full", experiment_name=None,
                        emb_cache=True, llm_timeout_s=300.0):
    """Main function to run reproduction-focused paper test.
    code_gen_variant: 'full' = paper + full expert graph; 'retrieval' = paper + retrieved expert snippets.
    experiment_name: if set, creates log_path/experiment_name/ for submission, code_generation, and log.
    llm_timeout_s: per-call wall-clock ceiling; on timeout the call raises so the chunk retry loop fires early.
    """

    # Experiment path: when set, submission, code_generation, and log go to log_path/experiment_name/
    output_base = log_path
    if experiment_name:
        experiment_path = os.path.join(output_base, experiment_name)
        code_output_dir = experiment_path
    else:
        code_output_dir = output_base
    
    base_url = os.getenv("OPENAI_API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL") # "Qwen/Qwen3-Next-80B-A3B-Instruct" #"Qwen/QwQ-32B"

    # Get paper directory and load repo URL
    paper_dir = os.path.dirname(paper_path)
    if not use_repo:
        repo_url = None
    else:
        repo_url = reprutil.load_repo_url(paper_dir)
        
    log(f"Paper: {paper_path}")
    log(f"Repository: {repo_url or 'Not specified'}")
    log(f"Model: {model}")
    log(f"Device: {device}")
    log("")

    total_start_time = time()

    paper_graph = ReproductionGraph(model, 
    "You are a helpful assistant specializing in research reproduction",
                            api_key, log, base_url, device, type='paper')
    mem_graph = ReproductionGraph(model,
    "You are a helpful assistant specializing in research replication",
                                api_key, log, base_url, device, type='mem')
    paper_graph.emb_cache = emb_cache
    mem_graph.emb_cache = emb_cache

    # Bound per-call wall-clock. Without this a stalled (queued on the shared GPUs)
    # or repetition-looping generation never raises, so the chunk-extraction retry
    # loop below can't fire and a single call blocks for many minutes. With a timeout
    # the call raises promptly and the existing retry kicks in early.
    paper_graph.client = paper_graph.client.with_options(timeout=llm_timeout_s)
    mem_graph.client = mem_graph.client.with_options(timeout=llm_timeout_s)
    log(f"LLM per-call timeout: {llm_timeout_s}s")

    if os.path.exists(
        os.path.join(output_base, "paper_graph_data.json")) and os.path.exists(
            os.path.join(output_base, "mem_graph_data_with_code.json")):
        log("Loading paper and mem graphs from disk") 
        paper_graph.load_triplets_from_json(os.path.join(output_base, "paper_graph_data.json"),clear_first=True,)
        mem_graph.load_triplets_from_json(os.path.join(output_base, "mem_graph_data_with_code.json"),clear_first=True,)
        
    else:
        log("Loading paper...")
        paper_content = reprutil.load_paper(paper_path)
        sections = reprutil.split_into_sections(paper_content)
        log(f"Paper split into {len(sections)} sections")
        log("")
        # Process sections
        section_stats = []

        for i, section in enumerate(sections):
            #if "references" in section['title'].lower():
            #    break
            section_start_time = time()
            current_datetime = datetime.datetime.now()
            log("="*70)
            log(f"{current_datetime} SECTION {i+1}/{len(sections)}: {section['title']}")
            log("="*70)

            chunks = reprutil.preprocess_section(section['content'])
            log(f"Section split into {len(chunks)} chunks")

            section_triplets_paper = []
            section_triplets_mem = []
            for j, chunk in enumerate(chunks):
                log(f"\n{datetime.datetime.now()} Processing chunk {j+1}/{len(chunks)}...")
                log(f"Chunk preview: {chunk[:100]}...")

                
                retries = 3
                while retries > 0:
                    try:
                        log(f"Try {4-retries}/{retries}")
                        new_triplets_paper, _ = paper_graph.update_without_retrieve(chunk, [], log, source_type="paper")
                        section_triplets_paper.extend(new_triplets_paper)
                        new_triplets_mem = None
                        
                        new_triplets_mem, _ = mem_graph.update_without_retrieve(chunk, [], log, source_type="paper")
                        section_triplets_mem.extend(new_triplets_mem)
                        log(f"{datetime.datetime.now()} Chunk {j+1}/{len(chunks)} done")
                        if new_triplets_paper:
                            log("Sample triplets paper:")
                            for triplet in new_triplets_paper[:3]:
                                subj, obj, rel = triplet
                                log(f"  - ({subj}) --[{rel.get('label', 'N/A')}]--> ({obj})")
                        
                        if new_triplets_mem:
                            log("Sample triplets mem:")
                            for triplet in new_triplets_mem[:3]:
                                subj, obj, rel = triplet
                                log(f"  - ({subj}) --[{rel.get('label', 'N/A')}]--> ({obj})")
                        break
                    except Exception as e:
                        log(f"Error processing chunk: {str(e)}")
                        retries -= 1
                        if retries > 0:
                            log(f"Retrying ({retries} left)...")
                            sleep(5)
                        else:
                            raise e

            section_time = time() - section_start_time

            log("")
            log(f"Section summary:")
            log(f"  - Triplets extracted: paper {len(section_triplets_paper)}, mem {len(section_triplets_mem)}")
            log(f"  - Total triplets in graph: paper {len(paper_graph.triplets)}, mem {len(mem_graph.triplets)}")
            log(f"  - Processing time: {section_time:.2f} seconds")
            log(f"  - API tokens: paper {paper_graph.completion_tokens} completion, {paper_graph.prompt_tokens} prompt")
            log(f"  - API tokens: mem {mem_graph.completion_tokens} completion, {mem_graph.prompt_tokens} prompt")
            log("")
            # Save current graph state after each section (to output_base for shared access)
            reprutil._write_json_to_path(output_base, "paper_graph_data.json", {
                "paper": os.path.basename(paper_path),
                "sections_processed": i + 1,
                "triplets": [[t[0], t[1], t[2]] for t in paper_graph.triplets],
                "obs_episodic": {k: v[0] for k, v in paper_graph.obs_episodic.items()},
                "stats": {"total_triplets": len(paper_graph.triplets), 
                "prompt_tokens": paper_graph.prompt_tokens,
                "completion_tokens": paper_graph.completion_tokens},
                
            })
            log("Graph state saved to paper_graph_data.json")
            reprutil._write_json_to_path(output_base, "mem_graph_data_with_code.json", {
                "paper": os.path.basename(paper_path),
                "sections_processed": i + 1,
                "triplets": [[t[0], t[1], t[2]] for t in mem_graph.triplets],
                "obs_episodic": {k: v[0] for k, v in mem_graph.obs_episodic.items()},
                "stats": {"total_triplets": len(mem_graph.triplets), 
                "prompt_tokens": mem_graph.prompt_tokens,
                "completion_tokens": mem_graph.completion_tokens},
            })
            log("Graph state saved to mem_graph_data_with_code.json")

            section_stats.append({
                'section_num': i + 1,
                'title': section['title'],
                'triplets_extracted_paper': len(section_triplets_paper),
                'triplets_extracted_mem': len(section_triplets_mem),
                'processing_time': section_time,
                'triplets_paper': [[t[0], t[1], t[2]] for t in section_triplets_paper],
                'triplets_mem': [[t[0], t[1], t[2]] for t in section_triplets_mem],
            })

        # Persist embedding sidecars next to the freshly extracted graphs so the next
        # load (resume here, or continual reusing this paper graph) skips re-embedding.
        if emb_cache:
            save_graph_embeddings(os.path.join(output_base, "paper_graph_data.json"), paper_graph, log)
            save_graph_embeddings(os.path.join(output_base, "mem_graph_data_with_code.json"), mem_graph, log)

    #'''
    # =============================================================================
    # PHASE 2: Code to mem graph Linking
    # =============================================================================
    if len(mem_graph.triplet2code) == 0:
        log("Linking code to  MEMORY graph...")

        if repo_dir is not None or repo_url is not None:
            log("="*70)
            log("PHASE 2: CODE INDEX + PAPER-TO-CODE LINKING")
            log("="*70)

            if repo_dir is not None:
                temp_dir = repo_dir
                log(f'Using local repo at {temp_dir}')
            else:
                assert False, "No repo provided."
            
            code_files = reprutil.collect_code_files(temp_dir)
            log(f"Found {len(code_files)} code files")

            log("\nBuilding code index...")
            #code_index = reprutil.build_code_index(temp_dir, code_files, log)
            #log(f"Code index: {len(code_index)} entities (classes, functions, config keys)")

            log("\nRunning linking pass (hypernodes: patterns -> code)...")
            link_start = time()
            mem_graph = run_linking_pass_hypernodes(
                mem_graph, code_files, log,
                output_base=output_base, paper_path=paper_path, repo_dir=temp_dir,
            )
            link_time = time() - link_start
            log(f"Linking complete: {len(mem_graph.triplet2code)} hypernodes in {link_time:.2f}s")
            log(f"Total cost so far: ${mem_graph.total_amount:.4f}")
            log('Saving graph with code linking to a file0')
            reprutil._write_json_to_path(output_base, "mem_graph_data_with_code.json", {
                "paper": os.path.basename(paper_path),
                "triplets": [[t[0], t[1], t[2]] for t in mem_graph.triplets],                           
                "obs_episodic": {k: v[0] for k, v in mem_graph.obs_episodic.items()},
                "triplet2code": mem_graph.triplet2code,
                "stats": {"total_triplets": len(mem_graph.triplets), 
                "prompt_tokens": mem_graph.prompt_tokens,
                "completion_tokens": mem_graph.completion_tokens},
            })
        else:
            log('No repo provided, skipping Phase 2 (code linking).')

    
    
    # =============================================================================
    # PHASE 3: Generate and validate code from graph
    # =============================================================================
    code_generation = {}
    if generate_code:
        log("="*70)
        log("PHASE 3: GRAPH-ONLY CODE GENERATION")
        log("="*70)
        
        #if code_gen_variant == "retrieval":
        code_generation = generate_code_from_graph_with_retrieval(
            mem_graph=mem_graph,
            paper_graph=paper_graph,
            log=log,
            output_dir=code_output_dir if experiment_name else None,
            graph_base_dir=output_base,
            #retrieval_topk=3,
            #retrieval_max_depth=2,
            #retrieval_threshold=0.7,
            #max_paper_queries=25,
            code_snippet_len=10000,
        )
        #else:
        #    code_generation = generate_code_from_graph(
        #        graph=graph,
        #        paper_path=paper_path,
        #        paper_name=args.paper,
        #        log=log,
        #        output_dir=code_output_dir if experiment_name else None,
        #        graph_base_dir=output_base,
        #    )
        #if test_generated_code and code_generation.get("code_path"):
        
        #exec_results = test_generated_code_executability(code_generation["code_path"], log)
        #code_generation["executability"] = exec_rreprutil.estimate_tokens(prompt, model=os.getenv("OPENAI_MODEL"))n.json")
        log("code_generation saved to code_generation.json")
    else:
        log("Skipping graph-based code generation (--no-generate-code)")
       
    

    log("="*70)
    log("REPRODUCTION RUN COMPLETE")
    log("="*70)
    log(f"Total time: {time() - total_start_time:.2f} seconds")
    log("")


def main():
    parser = argparse.ArgumentParser(
        description='Build replication-focused cookbook knowledge graph from academic paper'
    )
    parser.add_argument('--paper', type=str,
                        default='adaptive-pruning',
                        help='Path to paper markdown file')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for retriever (default: cpu)')
    parser.add_argument('--log-path', type=str, default='',
                        help='Path for log output (default: none)')
    parser.add_argument('--generate-code', dest='generate_code', default=True, type=bool,
                        help='Generate code from final graph (default: enabled)')
    #parser.add_argument('--test-generated-code', dest='test_generated_code', default=True, type=bool,
    #                    help='Run executability tests for generated code (default: enabled)')
    parser.add_argument('--use-repo', dest='use_repo', default=False, type=bool,
                        help='use repo url from blacklist.txt')
    parser.add_argument('--repo-dir', type=str, default=None,
                        help='Path to local repository (use instead of cloning)')
    parser.add_argument('--code-gen-variant', type=str, default='full',
                        choices=['full', 'retrieval'],
                        help='Code generation variant: full=paper+full expert graph, retrieval=paper+retrieved expert snippets (default: full)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name inside output_dir: creates output_dir/experiment/ with submission, code_generation, and log (default: none)')
    parser.add_argument('--emb-cache', dest='emb_cache', type=str2bool, default=True,
                        help='Cache triplet embeddings to a _emb.npz sidecar to skip re-embedding on reload (default: true). Disable with --emb-cache false.')
    parser.add_argument('--llm-timeout-s', dest='llm_timeout_s', type=float, default=300.0,
                        help='Per-call LLM wall-clock ceiling in seconds (default: 300). On timeout the call raises so the chunk-extraction retry loop fires early instead of blocking. Loosen with --llm-timeout-s 600.')

    args = parser.parse_args()
    paperbench_data_path = '/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/'
    # '/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/short-papers/'
    
    output_base = args.log_path + '/' + args.paper
    if args.experiment:
        experiment_path = os.path.join(output_base, args.experiment)
        os.makedirs(experiment_path, exist_ok=True)
        log = Logger(experiment_path)
        log(f"Experiment: {args.experiment} (output in {experiment_path})")
    else:
        log = Logger(output_base)
        log(f"Output in {output_base}")

    try:
     
        run_reproduction_test(
            log=log,
            args=args,
            paper_path=paperbench_data_path + args.paper + '/paper.md',
            device=args.device,
            log_path=args.log_path + '/' + args.paper,
            generate_code=args.generate_code,
            use_repo=args.use_repo,
            repo_dir=args.repo_dir,
            code_gen_variant=args.code_gen_variant,
            experiment_name=args.experiment,
            emb_cache=args.emb_cache,
            llm_timeout_s=args.llm_timeout_s,
        )

    except Exception as e:
        log("Unexpected Exception: %s" % str(e))
        raise e
        

if __name__ == "__main__":
    main()
