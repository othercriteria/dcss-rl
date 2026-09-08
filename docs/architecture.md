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
DCSS is ready for more input. Their preferred boundary is the upstream `input_mode=1`
signal followed by a flush; the transport remembers readiness across batches because
DCSS emits mode changes rather than repeating the current mode. If that signal is
absent, automatic commands retain a conservative command-specific quiescence fallback
with an explicit `Seconds` type.

For ordinary actions, silence is not itself an error: 0.33.1 can process WAIT without
emitting any visible delta or flush. If no output appears promptly, the adapter queues
upstream's `spectator_joined` full-state request behind the key. DCSS handles that
request on its next input-loop entry, providing a concrete synchronization response
without modifying game state or upstream source. Automatic travel and rest may poll
control messages while still running, so they use the input-ready signal rather than
misusing this probe as an end-of-travel barrier. Adversarial transport tests cross an
intermediate flush before accepting readiness and cover the quiescence fallback. Raw
control and response messages remain in the trajectory for later audit.

Local agent RC files disable view, travel, rest, and monster-in-sight animations using
ordinary documented DCSS options. These are presentation sleeps only: they do not
alter game turns or rules. This matters particularly when autoexplore repeatedly stops
on a visible monster; trunk otherwise sleeps 100 ms for every warning flash.

## Observations

WebTiles sends sparse deltas rather than full states. The reducer mirrors these stable
protocol behaviors:

- player fields merge recursively;
- map cells omit coordinates when they follow the previous cell horizontally;
- `clear: true` resets remembered map cells;
- inventory entries with zero quantity are omitted from semantic output;
- presentation markup is removed from policy-facing messages;
- menus expose their visible prompt and offered hotkeys;
- both widget-style `ui-push` menus and structured `menu` prompts expose their visible
  titles and every accepted hotkey;
- unknown fields remain available in raw trajectories even when not promoted to the
  semantic schema.

The reducer caches normalized semantic cells behind a frozen domain type and
invalidates their deterministic ordering on map updates. Every emitted cell view is
still newly materialized (including nested monster data), so callers cannot mutate a
past or future snapshot through the cache.

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

Trajectory schema v2 stores one initial semantic snapshot and a minimal patch per
transition: changed/removed player fields, changed/removed map cells, current messages,
and explicit menu/input-mode changes. Policy-action and next-delta segments retain
separate loss labels. Exact raw messages remain alongside the delta, and applying each
patch reconstructs the policy-facing snapshots required as future training context.

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

The initial `watch-best` implementation is a player-centered terminal replay. It reads
the champion manifest, defaults to its first suite-ordered episode, supports explicit
case IDs, and reconstructs both schema-v1 and schema-v2 trajectories. A native live
spectator can be added without changing champion selection or replay semantics.

The checked-in diagnostic suite is used for policy and adapter development. Once a
diagnostic seed has informed a code change it cannot be called held out. The held-out
manifest therefore contains a disjoint, untouched seed set and is used only for
champion evaluation. Independent cases execute concurrently; deterministic result
ordering follows manifest order rather than completion order.

The canonical learned held-out rank is also checked in as a regression floor. The
standard scripted evaluation command verifies that floor before writing a champion
manifest. Threshold comparison uses the same lexicographic metric ordering as
champion selection, so improvement on a higher-priority milestone is not vetoed by a
lower-priority aggregate. Rank v5 orders wins, runes, depth-weighted newly discovered
cells, distinct `(place, branch depth)` levels visited, aggregate maximum depth,
maximum XL, then reward. Already-seen cells, repeated actions, and lingering add no
credit; newly exploring after a tactical or strategic retreat does. Death preserves
progress accumulated before reset instead of forfeiting an artificial remainder of
the evaluation horizon. XL is monotonic in normal play, so integrating it over
decisions would reward risky front-loading rather than additional progression. The
rank excludes both policy decisions and raw DCSS turns as survival proxies.

## Initial research comparisons

1. Transparent scripted baseline.
2. Conventional recurrent policy-gradient baseline.
3. Language-model policy-gradient baseline.
4. Matched language-model policy plus ECHO-style environment prediction.

The RTX 4090 favors small models and parameter-efficient tuning initially. The
environment, trajectory format, evaluation, and viewer must not depend on a particular
trainer implementation.

## First learned baseline

The first falsifiable learner is a fixed-width semantic actor/value network rather
than an LM. Its deterministic feature contract contains a player-centered local map,
normalized character statistics, prompt/message indicators, and global navigation
summaries computed only from remembered player-visible cells. A shared encoder feeds
masked action logits, discounted-return value regression, and an ECHO auxiliary head
that predicts the next semantic feature delta. The auxiliary weight is configurable
for later ablation.

Training begins with behavior cloning and scripted-relabel DAgger. The latter labels
learner-visited states with the transparent expert, directly addressing compounding
errors without using held-out games. Online DAgger aggregates learner-state labels
across updates; only its imitation minibatches use replay, while PPO/value/ECHO losses
remain on the current on-policy rollout. Checkpoints store a schema version, feature-spec
version, model configuration, plain state dictionary, policy identity, training
method/hyperparameters, source counts, and validation accuracy. Feature changes bump
the feature-spec version and fail closed when loading older checkpoints.

Action history is agent-side context, not an environment observation. History-aware
models concatenate a checkpointed, fixed number of newest-first one-hot action slots
at the encoder boundary; ECHO continues to predict only the semantic next-environment
delta. Evaluation owns one history per episode and rollout workers own one per game,
so concurrent cases cannot contaminate each other. A stateless checkpoint can be
expanded with zero history columns, preserving its initial function before online
DAgger/PPO learns history-dependent behavior.

The action catalog is append-only at the checkpoint boundary. The original command
and 256 menu-key indices remain fixed; later structured commands occupy new tail
indices. Loading an older checkpoint expands its policy head with zero weights and a
low initial bias for new actions, and remaps any per-slot history columns without
changing established indices. This permits adding player-visible multi-step UI flows,
such as opening Trog's ability menu and selecting Berserk, without invalidating prior
policies or encoding a privileged macro.

An optional staged warmup fits appended policy-head rows, declared dependent menu-key
rows, and newly appended semantic input columns. All established input columns,
hidden/value layers, and unrelated heads are restored after each optimizer step
(including AdamW decay), and their gradients are zeroed so optimizer moments cannot
accumulate invisibly. This supports multi-decision affordances without treating them
as privileged macros. Online DAgger may also preload replayable, current-teacher-
relabeled training trajectories; exact paths are checkpoint metadata. Later updates
may deliberately unfreeze the full network.

Feature v4 appends explicit player-visible Berserk and exhaustion indicators. Migration
zero-initializes their input and ECHO coordinates while preserving every legacy
policy/value output and legacy ECHO coordinate. Old checkpoints continue to evaluate
under their recorded feature version; online training upgrades them explicitly.

Diagnostic and held-out champion manifests are separate monotonic tracks. A candidate
can replace a track only when its metric vector strictly outranks the existing
same-suite manifest; cross-suite promotion is rejected.

PPO return estimation distinguishes scored episode boundaries from the continuing
training task. A time limit bootstraps the value of its final player-visible
observation. In optional `continuing-reset` mode, death instead bootstraps the value of
a freshly reset, independently scheduled training game; ascension remains terminal.
Both boundaries cut GAE recursion, so rewards from the next rollout slot cannot leak
backward. This makes death neither an explicit penalty nor a way to erase future
decision costs. Episodic zero-terminal behavior remains available as a matched control.

Training can charge independent typed costs for every policy decision and for returning
to an exact player-visible semantic state within a bounded decision window. The cycle
fingerprint retains absolute position, player statistics/status, visible map, menu, and
input mode while excluding transient messages and DCSS turn/time clocks. Thus ordinary
repeated action labels are not presumed unsafe, and evaluation rewards/ranks remain
unchanged. Cycle incidence and all cost settings are recorded with update telemetry and
checkpoint metadata.

Experimental conclusions are conditional on checkpoint, representation, curriculum,
objective weights, and budget. A failed arm controls the next local decision; it is
not evidence that its knob is permanently useless after those surrounding conditions
change.

Long online runs may retain an immutable checkpoint after every completed update in
addition to the atomically replaced final checkpoint. Intermediate policy selection
uses only the known diagnostic suite; heldout access remains a single final promotion
gate. Naturally sorted update names make the training curve reproducible without
inferring state from modification times.

The first promoted agent uses a confidence gate calibrated only on diagnostic expert
states. At threshold 0.98, neural decisions covered 23.7% of diagnostic actions with
100% agreement on the calibration set; the transparent expert handles lower-confidence
states. Evaluation reports the realized learned-action fraction so hybrid gains cannot
be mistaken for fully autonomous neural control. Increasing that fraction while
retaining depth is the next learning objective.

Online PPO collection assigns each worker an independent episode chunk so slow games
do not impose a per-action barrier. Workers submit encoded states to one inference
service, which waits a typed, bounded interval to coalesce GPU requests while game
processes continue independently. Every worker occupies a stable row in a matrix padded
to the configured worker count. This spends cheap GPU capacity to hold both matrix
shape and row position constant: asynchronous scheduling therefore cannot perturb
sampled actions through batch-shape-dependent floating-point rounding. The model is
immutable during each rollout and is
updated only after all chunks complete. Worker-local random generators preserve action
sampling independence; fixed-suite repeats must remain tensor-identical despite
request scheduling. Checkpoint metadata records the inference batch bound and wait,
while update telemetry separates collection, optimization, and persistence time.

Feature encoding is carried across adjacent rollout decisions and offline trajectory
transitions rather than recomputed from the same immutable semantic observation. At a
continuing death boundary, the terminal encoding remains the ECHO next-environment
target while the separately encoded reset state is cached for the next policy decision
and value bootstrap. Default-disabled reward shaping bypasses its full-map potential
scan, and Gym materializes the returned current observation only once.

Training worker directories use only typed worker, episode, and startup-attempt indices;
human-controlled suite case labels never contribute to the Unix socket path. Evaluation
keeps case labels at the outer artifact layer but gives each transient startup attempt
its own trajectory and game directory. Both paths retry startup timeouts three times,
preserving failed-attempt evidence rather than overwriting it.

Gymnasium requires overridden reset/step methods to expose an invariant
`dict[str, Any]` info type. That third-party seam delegates immediately to
`reset_typed` and `step_typed`; project code uses those typed methods and carries
`EnvironmentInfo`, `ResetOptions`, `ActionIndex`, and `GameSeed` without raw-container
or primitive leakage.

On a fixed five-case, 50-decision learned-policy workload, throughput scales from
2.96 decisions/s with one worker to 5.55 with two and 12.13 with five. All worker
counts produce the identical metric vector and learned-action fraction. This makes
additional rollout breadth comparatively cheap; longer horizons remain independently
valuable because early-floor behavior underrepresents later-level tactical states.
