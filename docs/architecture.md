# MVP architecture and experimental contract

## Boundary

DCSS runs as an unmodified subprocess built from the official source. Each episode has
an isolated mutable directory. A direct Unix datagram client speaks the same protocol
as the official WebTiles server, avoiding browser/account overhead without adding a
privileged game API.

```text
policy -> structured action -> deterministic key encoder -> DCSS
                                                     |
policy <- semantic snapshot <- stateful reducer <- WebTiles deltas
                                      |
                               raw trajectory log
```

DCSS fragments large JSON messages into datagrams and terminates each logical message
with a newline. A starred message is server control. In particular,
`*{"msg":"flush_messages"}` marks the point at which the emitted deltas form one
atomic rendering batch. Automatic travel and rest can emit several such batches before
DCSS is ready for more input, so the transport coalesces consecutive batches until a
short command-specific quiescence period. Multi-turn commands use a longer duration
than ordinary actions; both durations carry an explicit `Seconds` type.

## Observations

WebTiles sends sparse deltas rather than full states. The reducer mirrors these stable
protocol behaviors:

- player fields merge recursively;
- map cells omit coordinates when they follow the previous cell horizontally;
- `clear: true` resets remembered map cells;
- inventory entries with zero quantity are omitted from semantic output;
- presentation markup is removed from policy-facing messages;
- menus expose their visible prompt and offered hotkeys;
- unknown fields remain available in raw trajectories even when not promoted to the
  semantic schema.

The semantic schema should favor names, glyphs, relative geometry, and ordinary game
statistics over version-specific tile IDs. Reducer changes must remain replayable over
previously collected raw messages.

## Actions

Structured actions are serialized intents lowered deterministically to normal player
inputs. The initial vocabulary is eight-way movement, wait, autoexplore, rest, stairs,
cancel, and selection among DCSS-provided menu hotkeys. Later targeted/item/god actions
may require multiple input boundaries and should be modeled as explicit state machines,
not opaque key strings.

Legality is deliberately narrow: an action is legal when it is valid for the visible
UI state. It does not imply success or tactical wisdom. Unknown input modes fail closed
to cancel-only behavior.

Gym uses a fixed discrete catalog: stable command actions followed by one menu-selection
slot for each byte-valued keycode. Each observation supplies a boolean mask over that
catalog. Policy-facing observations remain structured JSON-compatible dictionaries;
their variable map and inventory sizes are validated by a small custom Gym space.

Domain scalars are distinct types where accidental interchange is meaningful:
`GameSeed`, `ActionIndex`, `Keycode`, `StepLimit`, `WorkerCount`, and `Seconds` do not
silently cross application boundaries. Coordinates use a named tuple alias. Raw
primitives are retained only where Gym, JSON, subprocess, or socket APIs require them.

## Trajectories and ECHO

An episode record must be sufficient to reproduce, audit, and re-reduce a rollout:

- schema version, DCSS commit/version, build flags, and RC digest;
- character configuration, explicit seed when used, reward specification;
- agent/checkpoint identity and code revision;
- raw ordered WebTiles messages and flush boundaries;
- structured actions plus exact emitted keycodes;
- semantic snapshots and terminal outcome/morgue paths;
- tokenized training examples with separate policy-action and environment-observation
  masks.

ECHO-style training adds next-environment-token cross-entropy to a policy-gradient
objective. DCSS is stochastic and partially observable, so prediction quality is
probabilistic. Mask volatile boilerplate and report calibration; do not train primarily
on repeated full-screen payloads when compact semantic deltas suffice.

## Evaluation and champion

Training seeds and diagnostic scenarios are disjoint from held-out evaluation. Headline
evaluation uses ordinary unseeded games; deterministic held-out seeds may additionally
serve regression comparisons across agents and DCSS versions.

Report a metric vector rather than hiding all behavior in one score: ascensions, runes,
branch/depth progress, XL, turns survived, and deaths. A documented ordering selects
one champion manifest from a fixed suite. `watch-best` always runs that manifest (or a
clearly labeled scripted champion before learned checkpoints exist) and records the
selected seed policy so viewing cannot become cherry-picking.

The checked-in diagnostic suite is used for policy and adapter development. Once a
diagnostic seed has informed a code change it cannot be called held out. The held-out
manifest therefore contains a disjoint, untouched seed set and is used only for
champion evaluation. Independent cases execute concurrently; deterministic result
ordering follows manifest order rather than completion order.

## Initial research comparisons

1. Transparent scripted baseline.
2. Conventional recurrent policy-gradient baseline.
3. Language-model policy-gradient baseline.
4. Matched language-model policy plus ECHO-style environment prediction.

The RTX 4090 favors small models and parameter-efficient tuning initially. The
environment, trajectory format, evaluation, and viewer must not depend on a particular
trainer implementation.
