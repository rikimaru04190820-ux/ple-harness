# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plan-work-verify harness with MCP tools and context compaction.

Built from the `plan-verify` sample (itself a three-role split of `minimal`),
with the MCP discovery and compaction approaches of `custom-openai` ported
in, and constraint-first prompting throughout:

1. The planner inspects the task's files and writes a plan describing each
   deliverable, its path and format, how to verify it, and a "Hard
   constraints" section listing every mandatory requirement in the task.
2. The worker executes the task using the plan; its answer becomes the final
   message. It is instructed to satisfy every hard constraint before
   optimizing anything else and to re-check each one before finishing.
3. The verifier starts with a fresh conversation containing only the task,
   the plan, and the worker's summary. It checks the hard constraints first,
   then replies `PASS` or lists specific gaps, which are sent back to the
   worker for up to `MAX_FIX_ROUNDS` rounds while sufficient time remains.

All roles also get the task's own MCP tools (the planner and verifier only
the ones the server marks read-only): a small catalog is offered directly as
functions, a large one through `mcp_find`/`mcp_call`. Long conversations are
compacted before they overflow: older tool results are truncated first, then
older turns are replaced by the model's own notes; the truncate-on-400
recovery of the base samples remains as a fallback.
"""

import asyncio
import json
import os
import re

import openai
import taskbridge

MODEL = os.environ["MODEL_NAME"]
# The model's highest reasoning level: max for Claude, GLM, GPT-6 and
# DeepSeek-V4, high for the others (Gemini refuses max). REASONING_EFFORT
# overrides it.
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
# a fix round wraps up this long before the time limit, and a wrap-up call may
# take as long; the worker's final stage also reserves it, so verification
# can never consume the time needed to write the final deliverables
WRAP_UP_SEC = 240
WRAP_UP = {
    "role": "user",
    "content": "Time is up for this step. Stop and reply with your answer.",
}
RESULT_CHARS = 20_000  # a longer tool result loses its middle
# Seconds between retries once the SDK's own quick retries are exhausted.
RETRY_WAITS = (5, 15, 30, 60, 120)
# How endpoints word a 400 saying the conversation no longer fits the context
# window: Claude, OpenAI and vLLM, Gemini, xAI.
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1000  # characters a tool result keeps once the context is full
# times in a row an empty or cut-off reply gets a "continue" before the run
# stops
MAX_NUDGES = 3
# lower it for a small context window
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
# Claude caches a prompt prefix only when marked with cache_control. Each call
# marks the system prompt and the latest message (using 2 of Claude's 4 cache
# breakpoints) so each turn reads the prior conversation from cache. Other
# providers cache automatically without markers.
CACHE_MARK = "claude" in MODEL.lower()
# the planner wraps up once it has used this share of the task's time
PLAN_SHARE = 0.1
# the worker wraps up when this share is left, to keep time for checking and
# fixing
CHECK_SHARE = 0.25
# a check or a fix round starts only with this long to work before its own
# wrap-up
MIN_STEP_SEC = 120
MAX_FIX_ROUNDS = 2
# compaction (from custom-openai): prompt-token threshold, how many recent
# tool results truncation always keeps, and how many trailing messages the
# notes replacement always keeps
COMPACT_AT = int(os.environ.get("COMPACT_AT_TOKENS", "150000"))
KEEP_RECENT_RESULTS = 12
KEEP_TAIL = 6
# MCP tools are offered as functions up to this many characters of JSON in
# all; functions one request may offer, the built-in three included
DIRECT_MCP_CHARS = 40_000
MAX_TOOLS = 128

# The evaluator sets OPENAI_BASE_URL and OPENAI_API_KEY for every model.
# The SDK handles quick retries; recover() handles longer backoffs. Because
# calls are non-streaming, a max-effort response can take over 15 minutes
# before any bytes arrive, so the timeout matches the egress proxy's 30-minute
# idle limit.
client = openai.AsyncOpenAI(max_retries=5, timeout=1800)

PLANNER = """You plan an engineering task that another engineer will carry out in a Linux container with no internet \
access. Inspect the task's files with your tools (bash starts in {workdir}), but do not create or change any file. \
{mcp}Then reply with the plan: each deliverable with its exact path and format, what it must contain, the data and \
method to use, and how to check it. The plan must also have a section titled "Hard constraints" listing every \
requirement the task states as mandatory — exact deliverable paths and formats, numeric limits, required contents, \
anything stated with "must", "shall", "at least", "at most" or "exactly" — each quoted or closely paraphrased, with \
how it will be verified. If the task states no hard constraints beyond the deliverables themselves, say so in that \
section."""

WORKER = """You are an expert engineer working on a task in a Linux container with no internet access. Your tools \
act inside it, and bash starts in {workdir}. A plan for the task follows it; use it, and correct it where the files \
show it is wrong.{mcp}

Constraints come first. The plan's "Hard constraints" section lists what the task makes mandatory: exact \
deliverable paths and formats, numeric limits, required contents. Satisfy every hard constraint before optimizing \
anything else — a deliverable that misses one is largely wasted work, however good the rest is. Compute with code, \
not by hand. Write each deliverable exactly where and in the format the task asks, then open it again to check it. \
Before you finish, re-read the task and go through the hard constraints one by one, verifying each against the \
actual files (recompute the numbers, reopen the files), and fix anything that fails. When you are done, reply \
without calling a tool: a short summary of each deliverable, where it is, and the key results."""

VERIFIER = """You check an engineer's work on a task in a Linux container. You get the task, the plan, and the \
engineer's summary of what was delivered. Open every deliverable with your tools (bash starts in {workdir}) and \
check it against the task.{mcp} Check the plan's "Hard constraints" first, one by one: each exact path, format, \
numeric limit and required content, recomputed from the files. Then check the rest: every part the task asks for, \
and the key numbers. Do not create or change any file. If everything holds, reply with the single word PASS and \
nothing else. Otherwise reply with a numbered list of the gaps, each specific enough to fix, hard-constraint \
failures first."""

MCP_DIRECT = """

The task also offers its own tools (the functions named <server>__<tool>); use them where the task's environment \
is the source of truth."""

MCP_CATALOG = """

The task also offers its own tools, listed below by server. Call mcp_find to see a tool's parameters, then \
mcp_call to run it.
{catalog}"""

SUMMARY_PROMPT = """The conversation is getting long, so the earlier turns will be removed. Write notes that let \
you continue without them: what the task requires (including every hard constraint), what you found in the files, \
every file you created or changed (with its path and state), the numbers and decisions so far, and what remains to \
do. Be specific."""


def tool(name, description, **params):
  """Builds an OpenAI function-tool definition dictionary.

  Args:
    name: Tool function name.
    description: Human-readable tool description.
    **params: Parameter names mapped to their JSON Schema type strings.

  Returns:
    An OpenAI function-tool specification dictionary.
  """
  return {
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": {k: {"type": v} for k, v in params.items()},
              "required": list(params),
          },
      },
  }


TOOLS = [
    tool(
        "bash",
        "Run a bash command (120 s limit) and return its exit code and output.",
        command="string",
    ),
    tool("read_file", "Read a text file.", path="string"),
    tool(
        "write_file",
        "Create or overwrite a file; parent folders are created.",
        path="string",
        content="string",
    ),
]
READ_ONLY = TOOLS[:2]

MCP_FIND = {
    "type": "function",
    "function": {
        "name": "mcp_find",
        "description": (
            "Show the description and parameters of the task's own tools"
            " that match a query."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
MCP_CALL = {
    "type": "function",
    "function": {
        "name": "mcp_call",
        "description": "Run one of the task's own tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "The tool's name as listed.",
                },
                "arguments": {"type": "object"},
            },
            "required": ["tool", "arguments"],
        },
    },
}


def clip(text):
  """Truncates `text` in the middle if it exceeds `RESULT_CHARS` characters.

  Args:
    text: Tool output string to clip.

  Returns:
    The original string if within `RESULT_CHARS`, or a middle-truncated string.
  """
  if len(text) <= RESULT_CHARS:
    return text
  half = RESULT_CHARS // 2
  return (
      f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut"
      f" ...]\n{text[-half:]}"
  )


def marked(messages):
  """Returns the messages a model call sends.

  For Claude, this is a copy whose system prompt and newest message carry
  `cache_control`. The stored conversation stays as it is, so each turn marks
  its own newest message.

  Args:
    messages: The conversation so far, in OpenAI chat format.

  Returns:
    `messages` itself for other models, or the marked copy for Claude.
  """
  if not CACHE_MARK:
    return messages
  out = list(messages)
  for i in {0, len(out) - 1}:
    m, content = out[i], out[i].get("content")
    if isinstance(content, str) and content:
      out[i] = {
          **m,
          "content": [{
              "type": "text",
              "text": content,
              "cache_control": {"type": "ephemeral"},
          }],
      }
    else:
      out[i] = {**m, "cache_control": {"type": "ephemeral"}}
  return out


def unique_name(name, taken):
  """Truncates `name` to 64 chars and appends `_2`, `_3`, ... if in `taken`.

  Args:
    name: Desired function name.
    taken: Collection or mapping of already-used function names.

  Returns:
    A unique function name of at most 64 characters.
  """
  candidate, n = name[:64], 1
  while candidate in taken:
    n += 1
    candidate = f"{name[:63 - len(str(n))]}_{n}"
  return candidate


class MCPServer(taskbridge.MCPClient):
  """Client wrapper for a single task MCP server and its discovered tools."""

  def __init__(self, session, spec):
    super().__init__(session, spec["url"], client="ple-harness")
    self.name = spec["name"]
    self.tools = []

  async def start(self):
    """Initializes the MCP session and fetches the server's tool list."""
    await self.initialize()
    self.tools = await self.list_tools()

  async def call(self, name, arguments):
    """Calls tool `name` on the MCP server and formats the result as text.

    Args:
      name: Unprefixed MCP tool name on this server.
      arguments: Tool input arguments dictionary.

    Returns:
      Formatted text output (prefixed with `error: ` if `isError` is set).
    """
    result = await self.call_tool(name, arguments)
    parts = []
    content = (
        result.get("content") if isinstance(result.get("content"), list) else []
    )
    for item in content:
      kind = item.get("type") if isinstance(item, dict) else None
      if kind == "text":
        parts.append(str(item.get("text", "")))
      elif kind == "resource":
        parts.append(json.dumps(item.get("resource"))[:RESULT_CHARS])
      elif kind == "resource_link":  # its uri, name and mimeType
        parts.append(json.dumps(item)[:RESULT_CHARS])
      else:
        parts.append(f"[{kind or 'unknown'} content not shown]")
    if not parts and result.get("structuredContent") is not None:
      parts.append(json.dumps(result["structuredContent"]))
    text = "\n".join(parts) or "(no content)"
    return f"error: {text}" if result.get("isError") else text


class MCP:
  """The task's MCP servers, connected once, with their tools registered.

  A small catalog is offered directly as OpenAI functions; a larger one is
  listed in the system prompt and reached through `mcp_find`/`mcp_call`.
  Read-only roles (planner, verifier) only ever receive the tools a server
  annotates `readOnlyHint`, plus `mcp_find` in catalog mode.
  """

  def __init__(self):
    self.registry = {}  # function name -> (MCPServer, tool dict)
    self.direct = []  # function definitions offered directly
    self.read_only_names = set()
    self.catalog = False

  async def connect(self, session):
    """Connects to the task's MCP servers and registers their tools.

    Args:
      session: Active `taskbridge.Session` holding `mcp_servers`.

    Returns:
      The system-prompt section describing the available MCP tools (empty
      when the task offers none or all are unreachable).
    """
    for spec in session.mcp_servers:
      server = MCPServer(session, spec)
      try:
        await server.start()
      except (taskbridge.TaskError, RuntimeError, ValueError) as exc:
        await session.log(f"MCP server {server.name} unavailable: {exc}")
        continue
      for t in server.tools:
        name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server.name}__{t['name']}")
        name = unique_name(name, self.registry)
        self.registry[name] = (server, t)
        if (t.get("annotations") or {}).get("readOnlyHint"):
          self.read_only_names.add(name)
      await session.log(
          f"MCP server {server.name}: {len(server.tools)} tools"
      )
    if not self.registry:
      return ""
    functions = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": (t.get("description") or t["name"])[:1024],
                "parameters": (
                    {
                        k: v
                        for k, v in (t.get("inputSchema") or {}).items()
                        if k != "$schema"
                    }
                    or {"type": "object", "properties": {}}
                ),
            },
        }
        for name, (_, t) in self.registry.items()
    ]
    if (
        len(json.dumps(functions)) <= DIRECT_MCP_CHARS
        and len(TOOLS) + len(functions) <= MAX_TOOLS
    ):
      self.direct = functions
      return MCP_DIRECT
    self.catalog = True
    servers = {}
    for name, (server, _) in self.registry.items():
      servers.setdefault(server.name, []).append(name)
    return MCP_CATALOG.format(
        catalog="\n".join(
            f"{s}: {', '.join(names)}" for s, names in servers.items()
        )
    )

  def tools(self, base, read_only=False):
    """Returns `base` plus the MCP tool definitions a role may use.

    Args:
      base: The role's built-in tool definitions.
      read_only: Restrict to non-mutating tools (planner/verifier).

    Returns:
      A new list of OpenAI function-tool definitions.
    """
    out = list(base)
    if not self.registry:
      return out
    if self.catalog:
      out.append(MCP_FIND)
      if not read_only:
        out.append(MCP_CALL)
      return out
    for fn in self.direct:
      if read_only and fn["function"]["name"] not in self.read_only_names:
        continue
      out.append(fn)
    return out

  def find(self, query):
    """Finds MCP tools matching `query` and formats their schemas.

    Returns as many matching tool schemas as fit within `RESULT_CHARS`
    (always at least one, without truncating any individual schema).

    Args:
      query: Tool name or search keywords.

    Returns:
      Formatted descriptions and JSON parameter schemas of matching tools.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", query.lower()) if w]

    def score(item):
      name, (_, t) = item
      text = f"{name} {t.get('description') or ''}".lower()
      return (
          (query == name) * 100
          + sum(w in name.lower() for w in words) * 3
          + sum(w in text for w in words)
      )

    best = [
        item
        for item in sorted(self.registry.items(), key=score, reverse=True)[:8]
        if score(item) > 0
    ]
    if not best:
      return (
          f"no tool matches {query!r}; the names are listed in the system"
          " prompt"
      )
    found = [
        f"{name}: {(t.get('description') or '')[:400]}\nparameters:"
        f" {json.dumps(t.get('inputSchema') or {})}"
        for name, (_, t) in best
    ]
    while len(found) > 1 and len("\n\n".join(found)) > RESULT_CHARS:
      found.pop()
    return "\n\n".join(found)

  async def call(self, name, arguments):
    """Runs the registered MCP tool `name` and returns its clipped output.

    Args:
      name: Function name as registered (e.g. `<server>__<tool>`).
      arguments: Tool input arguments dictionary.

    Returns:
      The clipped tool output or error string.
    """
    server, t = self.registry[name]
    return clip(await server.call(t["name"], arguments))


async def run_tool(session, mcp, fn, allowed):
  """Executes a single tool call and returns its output.

  Rewrites `fn["arguments"]` as canonical JSON (or `"{}"` if invalid) so the
  conversation history remains valid for strict endpoints. Calls to tools
  the current role was not offered are refused, so the planner and verifier
  cannot modify files or call mutating MCP tools. Any tool failure is
  returned as an error string for the model.

  Args:
    session: Active `taskbridge.Session` connected to the task container.
    mcp: The session's `MCP` registry.
    fn: Function-call dictionary containing `name` and `arguments`.
    allowed: Names of the tools offered to the calling role.

  Returns:
    The clipped tool output or error string to send back to the model.
  """
  name = fn["name"]
  try:
    args = json.loads(fn["arguments"] or "{}")
    if not isinstance(args, dict):
      raise TypeError("not a JSON object")
    fn["arguments"] = json.dumps(args, allow_nan=False)
  except (TypeError, ValueError) as exc:
    fn["arguments"] = "{}"
    return (
        f"error: the arguments are not a JSON object ({exc}); send the call"
        " again"
    )
  if name not in allowed:
    return f"error: the tool {name} is not available in this step"
  try:
    if name == "bash":
      result = await session.exec(args["command"])
      return clip(f"exit code {result['exit_code']}\n{result['output']}")
    if name == "read_file":
      return clip(
          (await session.read_file(args["path"])).decode("utf-8", "replace")
      )
    if name == "write_file":
      return (
          f"wrote {await session.write_file(args['path'], args['content'])}"
          f" bytes to {args['path']}"
      )
    if name == "mcp_find":
      return mcp.find(str(args.get("query", "")))
    if name == "mcp_call":
      target = str(args.get("tool", ""))
      arguments = args.get("arguments") or {}
      if isinstance(arguments, str):
        arguments = json.loads(arguments)
      if target not in mcp.registry:
        return f"error: no tool named {target!r}; use mcp_find"
      return await mcp.call(target, arguments)
    if name in mcp.registry:
      return await mcp.call(name, args)
    return f"error: there is no tool named {name}"
  # ValueError: content with an unpaired surrogate; RuntimeError covers
  # taskbridge.MCPError
  except (taskbridge.TaskError, KeyError, ValueError, RuntimeError) as exc:
    return f"error: {type(exc).__name__}: {exc}"


async def recover(session, messages, exc, waits, until):
  """Handles a failed model call and returns whether to retry.

  On a 400 context-overflow error, shortens the oldest tool results. On a
  rate limit, server error, or connection drop, sleeps for the next interval
  in `RETRY_WAITS` (capped at the step's wrap-up deadline). Errors are logged
  rather than added to the conversation.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, shortened in place when the context is full.
    exc: The `openai.APIError` the call raised.
    waits: Iterator over the remaining intervals in `RETRY_WAITS`.
    until: Seconds before the time limit at which the step wraps up.

  Returns:
    `True` to retry the call, `False` to re-raise.
  """
  status = getattr(exc, "status_code", None)  # None: no HTTP answer
  if status == 400 and CONTEXT_FULL.search(str(exc)):
    return await shorten(session, messages)
  wait = next(waits, None)
  if wait is None or not (
      status is None or status in (408, 409, 429) or status >= 500
  ):
    return False
  left = session.time_left()
  if left is not None and left > until:
    wait = min(wait, left - until)
  await session.log(
      f"model call failed ({str(exc)[:200]}); again in {wait:.0f} s"
  )
  await asyncio.sleep(wait)
  return True


async def shorten(session, messages):
  """Truncates the older half of long tool results to `CUT_KEEP` characters.

  Repeated calls truncate progressively newer results. Claude's thinking blocks
  are kept intact because each block's signature is self-contained and Claude
  rejects tool loops if the latest assistant turn loses its thinking block.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, modified in place.

  Returns:
    Whether any tool result was truncated.
  """
  long = [
      n
      for n, m in enumerate(messages)
      if m["role"] == "tool" and len(m["content"]) > 2 * CUT_KEEP
  ]
  if not long:
    return False
  cut = long[: max(1, len(long) // 2)]
  for n in cut:
    messages[n]["content"] = (
        messages[n]["content"][:CUT_KEEP]
        + "\n[... cut to save context; run the tool again if needed]"
    )
  await session.log(f"context full: cut {len(cut)} old tool results")
  return True


def estimate_tokens(messages):
  """Estimates the prompt size of `messages` in tokens (chars / 4 + overhead).

  Args:
    messages: The conversation so far, in OpenAI chat format.

  Returns:
    A rough token count, used only to decide when to compact.
  """
  chars = 0
  for m in messages:
    content = m.get("content")
    if isinstance(content, str):
      chars += len(content)
    elif isinstance(content, list):
      chars += sum(len(json.dumps(p)) for p in content)
    for call in m.get("tool_calls") or []:
      chars += len((call.get("function") or {}).get("arguments") or "")
  return chars // 4 + 8 * len(messages)


async def complete_once(session, messages):
  """Makes one tool-less model call and returns its text ("" on failure).

  Used for the compaction notes: the call carries no tools, reports its
  usage, and is cut off early enough to leave the step its wrap-up time.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation to send, in OpenAI chat format.

  Returns:
    The reply text, or an empty string when the call failed or timed out.
  """
  left = session.time_left()
  try:
    reply = await asyncio.wait_for(
        client.chat.completions.create(
            model=MODEL,
            messages=marked(messages),
            reasoning_effort=EFFORT,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        ),
        None if left is None else max(1.0, left - WRAP_UP_SEC),
    )
  except (openai.APIError, asyncio.TimeoutError) as exc:
    await session.log(f"context: notes call failed ({str(exc)[:200]})")
    return ""
  if reply.usage:
    details = getattr(reply.usage, "prompt_tokens_details", None)
    await session.usage(
        MODEL,
        reply.usage.prompt_tokens,
        reply.usage.completion_tokens,
        getattr(details, "cached_tokens", 0) or 0,
    )
  text = reply.choices[0].message.content or ""
  if reply.choices[0].finish_reason == "length" and text:
    text += "\n[the notes were cut off at the output limit]"
  return text


async def compact(session, messages, force=False):
  """Shrinks `messages` in place so the next model call fits the context.

  First truncates older tool results the model has already read (keeping
  the `KEEP_RECENT_RESULTS` most recent, and never the unread results of
  the latest turn). If that is not enough — or nothing could be cut —
  replaces the older turns with notes the model writes about its own work
  (`SUMMARY_PROMPT`), keeping the system prompt, the task, and the last
  `KEEP_TAIL` messages. Adapted from `custom-openai`.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, modified in place.
    force: Whether compaction was triggered by a context-full error; then
      all read results are cut and the notes are written whenever the
      estimate is still over the threshold.

  Returns:
    `True` if any tool results were truncated or turns were summarized.
  """
  last_assistant = max(
      (n for n, m in enumerate(messages) if m["role"] == "assistant"),
      default=-1,
  )
  read = [
      n
      for n, m in enumerate(messages)
      if m["role"] == "tool" and n < last_assistant
  ]
  cuts = 0
  for n in read if force else read[:-KEEP_RECENT_RESULTS]:
    content = messages[n]["content"]
    if isinstance(content, str) and len(content) > 600:
      messages[n]["content"] = (
          content[:400]
          + "\n[... removed to save context; run the tool again if needed"
          " ...]"
      )
      cuts += 1
  if cuts:
    await session.log(f"context: cut {cuts} old tool results")
    if not force or estimate_tokens(messages) <= COMPACT_AT:
      return True
  cut = len(messages) - KEEP_TAIL
  # the kept turns must not start with tool results whose call was replaced
  # by the notes
  while cut > 3 and messages[cut]["role"] == "tool":
    cut -= 1
  if cut < 10:  # too few turns to be worth a notes call
    return bool(cuts)
  notes = await complete_once(
      session, messages[:cut] + [{"role": "user", "content": SUMMARY_PROMPT}]
  )
  if not notes.strip():
    await session.log(
        "context: the notes came back empty, so the turns are kept"
    )
    return bool(cuts)
  messages[:] = (
      messages[:2]
      + [{
          "role": "user",
          "content": (
              "Your notes on the work so far (earlier turns were removed "
              "to save context):\n" + notes
          ),
      }]
      + messages[cut:]
  )
  await session.log(f"context: replaced {cut - 2} messages with notes")
  return True


def conversation(system, session, mcp_section, prompt):
  """Creates the initial `[system, user]` message list for a role.

  Args:
    system: System prompt template containing `{workdir}` and `{mcp}`.
    session: Active `taskbridge.Session` providing `workdir`.
    mcp_section: System-prompt section describing the task's MCP tools.
    prompt: Initial user message content.

  Returns:
    A two-element list of OpenAI chat message dictionaries.
  """
  return [
      {
          "role": "system",
          "content": system.format(
              workdir=session.workdir, mcp=mcp_section
          ),
      },
      {"role": "user", "content": prompt},
  ]


async def agent(session, mcp, messages, tools, until, final=False):
  """Runs the tool-calling loop on `messages` until the model answers.

  Prompts the model to wrap up immediately once fewer than `until` seconds
  remain on the task clock. Compacts the conversation first whenever the
  last call's prompt size passed `COMPACT_AT`. When `final` is `True`,
  updates `session.final_message` with each non-empty response.

  Args:
    session: Active `taskbridge.Session` for the task.
    mcp: The session's `MCP` registry, used to run MCP tool calls.
    messages: Mutable conversation history list.
    tools: List of OpenAI tool definitions available to this role.
    until: Remaining task time threshold (in seconds) at which this step
      wraps up.
    final: Whether to record responses in `session.final_message`.

  Returns:
    The final text response produced by this agent step.
  """
  allowed = {t["function"]["name"] for t in tools}
  prompt_tokens = 0  # prompt size of the previous call, from its usage
  # head: the start of an answer cut off at the output limit
  nudges, head, waits = 0, "", iter(RETRY_WAITS)
  while True:
    left = session.time_left()  # None when the task has no time limit
    wrap_up = left is not None and left < until
    if not wrap_up and prompt_tokens > COMPACT_AT:
      if await compact(session, messages):
        prompt_tokens = estimate_tokens(messages)
    if wrap_up and messages[-1:] != [WRAP_UP]:  # asked once, also on a retry
      messages.append(dict(WRAP_UP))
    # A call still running when the wrap-up starts is stopped. The wrap-up
    # call itself gets at most WRAP_UP_SEC and stops at the time limit; if
    # it takes longer, the step returns head, its answer so far.
    if left is None:
      timeout = None
    elif wrap_up:
      timeout = min(WRAP_UP_SEC, left)
    else:
      timeout = left - until
    try:
      reply = await asyncio.wait_for(
          client.chat.completions.create(
              model=MODEL,
              messages=marked(messages),
              tools=tools,
              tool_choice="none" if wrap_up else "auto",
              reasoning_effort=EFFORT,
              max_completion_tokens=MAX_OUTPUT_TOKENS,
          ),
          timeout,
      )
    except asyncio.TimeoutError:
      if wrap_up:
        return head
      continue
    except openai.APIError as exc:
      status = getattr(exc, "status_code", None)
      if (
          status == 400
          and CONTEXT_FULL.search(str(exc))
          and await compact(session, messages, force=True)
      ):
        prompt_tokens = estimate_tokens(messages)
        continue
      if await recover(session, messages, exc, waits, until):
        continue
      raise
    waits = iter(RETRY_WAITS)
    if reply.usage:
      details = getattr(reply.usage, "prompt_tokens_details", None)
      await session.usage(
          MODEL,
          reply.usage.prompt_tokens,
          reply.usage.completion_tokens,
          getattr(details, "cached_tokens", 0) or 0,
      )
      prompt_tokens = reply.usage.prompt_tokens or prompt_tokens
    message, cut = (
        reply.choices[0].message,
        reply.choices[0].finish_reason == "length",
    )
    text = head + (message.content or "")
    if final and text:
      session.final_message = text
    # sent back as returned: Gemini refuses a tool call without its thought
    # signature
    turn = message.model_dump(exclude_none=True)
    if wrap_up or not message.tool_calls:
      turn.pop("tool_calls", None)  # calls made at the wrap-up are not run
      if message.content:
        # a fix round, or the request to go on after a cut-off reply,
        # continues from this answer
        messages.append(turn)
      if wrap_up or (message.content and not cut) or nudges == MAX_NUDGES:
        return text
      nudges, head = nudges + 1, text
      messages.append({
          "role": "user",
          "content": (
              "Your reply was cut off; continue where it stopped."
              if cut
              else "Continue, and finish with your answer."
          ),
      })
      continue
    nudges, head = 0, ""
    messages.append(turn)
    for call in turn["tool_calls"]:
      await session.log(
          f"{call['function']['name']} {call['function']['arguments'][:200]}"
      )
      content = await run_tool(session, mcp, call["function"], allowed)
      messages.append(
          {"role": "tool", "tool_call_id": call["id"], "content": content}
      )
    if left is not None:
      messages[-1]["content"] += (
          f"\n[{int((session.time_left() - until) // 60)} minutes left for"
          " this step]"
      )


async def solve(session):
  """Executes the planner, worker, and verifier loop for `session`.

  Args:
    session: Active `taskbridge.Session` for the task.

  Returns:
    The final summary message from the worker.
  """
  total = session.time_left()

  def until(share):
    """Computes the remaining seconds when `share` of total task time is left.

    Args:
      share: Fraction of total task time remaining (`0.0` to `1.0`).

    Returns:
      Remaining seconds threshold, or `0` if the task has no time limit.
    """
    return 0 if total is None else total * share

  def time_for(wrap_up):
    """Checks whether at least `MIN_STEP_SEC` remains before `wrap_up`.

    Args:
      wrap_up: Remaining seconds threshold at which the next step wraps up.

    Returns:
      `True` if there is no deadline or at least `MIN_STEP_SEC` before
      `wrap_up`.
    """
    left = session.time_left()
    return left is None or left - wrap_up >= MIN_STEP_SEC

  mcp = MCP()
  section = await mcp.connect(session)
  task = session.instruction
  plan = await agent(
      session,
      mcp,
      conversation(PLANNER, session, section, task),
      mcp.tools(READ_ONLY, read_only=True),
      until(1 - PLAN_SHARE),
  )
  await session.log(f"plan: {plan[:1000]}")
  work = conversation(WORKER, session, section, f"{task}\n\n## Plan\n{plan}")
  summary = await agent(
      session, mcp, work, mcp.tools(TOOLS), until(CHECK_SHARE), final=True
  )
  for _ in range(MAX_FIX_ROUNDS):
    if not time_for(until(CHECK_SHARE / 2)):
      break
    review = (
        f"## Task\n{task}\n\n## Plan\n{plan}\n\n## The engineer's"
        f" summary\n{summary}"
    )
    verdict = await agent(
        session,
        mcp,
        conversation(VERIFIER, session, section, review),
        mcp.tools(READ_ONLY, read_only=True),
        until(CHECK_SHARE / 2),
    )
    await session.log(f"verifier: {verdict[:1000]}")
    # PASS alone in any letter case (Markdown emphasis and a full stop aside),
    # or an empty reply, names no gaps
    if verdict.strip("*_. \r\n\t").upper() in ("", "PASS") or not time_for(
        WRAP_UP_SEC
    ):
      break
    work.append({
        "role": "user",
        "content": (
            "A reviewer checked your files and found these"
            f" gaps:\n{verdict}\n\nFix them, then reply with your updated"
            " summary."
        ),
    })
    summary = (
        await agent(
            session, mcp, work, mcp.tools(TOOLS), WRAP_UP_SEC, final=True
        )
        or summary
    )
  return summary or session.final_message


if __name__ == "__main__":
  taskbridge.run(solve)
