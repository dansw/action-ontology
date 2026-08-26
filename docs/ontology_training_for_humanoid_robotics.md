# Contact-Oriented Ontologies for Humanoid Robot Training

## Thesis

Decomposing demonstration video into entities, resources, contacts, events,
and state transitions should provide a stronger learning signal for robotics
than raw video paired only with general descriptions.

A caption such as "the person catches an egg" captures the broad activity,
but omits details that matter for physical execution:

- which body resources participate;
- which surfaces touch the object;
- when contact begins and ends;
- which resource supports, grips, guides, or stabilizes the object;
- when an external entity becomes a controlled or held resource;
- what physical state changes as a result of the action.

Explicitly supervising these relationships reduces the amount of latent
physical structure a model must discover indirectly from pixels and coarse
language.

## Controlled-comparison verdict

Assume two systems have the same:

- model capacity and compute;
- sensors and observations;
- control and planning limitations;
- training-video coverage;
- geometry and dynamics limitations.

If one system receives raw video and general descriptions while the other
also receives accurate contact-oriented ontology supervision, the
ontology-enhanced system should theoretically perform better on tasks covered
by the training data.

Expected advantages include:

- **Higher sample efficiency:** explicit relationships reduce how much
  structure must be inferred independently from visual correlations.
- **Better task accuracy:** the model learns which resource acts on which
  object and which state transition constitutes success.
- **Better embodiment transfer:** functional roles such as support, grip,
  stabilize, push, rotate, and release can be assigned to the robot's own
  body and end effectors.
- **Better temporal understanding:** approach, contact initiation, control,
  possession, release, and post-action state become distinguishable events.
- **Better compositional behavior:** familiar resource-entity interactions
  can be recombined across related tasks.
- **Fewer visual shortcuts:** the model is less likely to identify a task
  solely from the scene, objects, or expected activity without grounding the
  actual interaction.
- **Better verification and debugging:** predicted contacts, actors, and
  state transitions can be inspected before and during execution.

The performance advantage should increase with task complexity and with the
importance of precise contact relationships.

## Tasks most likely to benefit

The ontology should provide the largest gains for:

- multistage object manipulation;
- bimanual coordination;
- grasp and handoff transitions;
- tool use;
- assembly and disassembly;
- deformable-object manipulation;
- tasks in which visually similar frames have different contact states;
- tasks where the same physical objective can be achieved through different
  human and robot embodiments.

The advantage may be smaller for simple locomotion, gross pose imitation, or
activities where general descriptions already contain nearly all relevant
task information.

## Embodiment-independent task representation

Directly copying human joint trajectories is fragile because a humanoid robot
may have different proportions, degrees of freedom, joint limits, hands, and
actuators. A more transferable representation captures the physical role of
each resource.

For example, a human demonstration might be represented as:

```text
left palm supports egg
left fingers constrain egg laterally
right fingers stabilize egg
```

The embodiment-independent constraints are:

```text
place a support surface below the object
provide lateral containment
add a secondary stabilizing contact
keep applied force below the object's damage threshold
```

A robot can assign those roles to its own palm, fingers, gripper surfaces, or
multiple manipulators. This separates *what physical relationships are
required* from *how this particular embodiment realizes them*.

## Recommended ontology structure

Entities, resources, and events are a useful foundation, but contact roles,
preconditions, outcomes, and uncertainty should also be represented.

```json
{
  "event": "establish_grasp",
  "actor_resources": ["left_palm", "left_fingers"],
  "entity": "egg",
  "contacts": [
    {"resource": "left_palm", "role": "support"},
    {"resource": "left_fingers", "role": "lateral_constraint"}
  ],
  "preconditions": ["egg descending", "hand open"],
  "postconditions": ["egg controlled by person", "egg velocity stabilized"],
  "state_transition": "external_entity -> held_resource",
  "confidence": 0.94
}
```

Important fields include:

- persistent identities for objects and resources;
- contact participants and contact roles;
- spatial relationships and support relationships;
- contact start and end times;
- object and resource states before and after each event;
- action preconditions and postconditions;
- confidence and visibility;
- distinctions between observation, inference, and intent.

## Recommended training architecture

The ontology should supplement the underlying observations rather than
replace them. The strongest training input combines:

```text
raw video
+ geometry and motion
+ structured contacts and events
+ state transitions
+ task outcomes
```

Useful synchronized supervision layers are:

1. Raw RGB video and, when available, depth, pose, tactile, and force data.
2. Persistent object and body-resource tracks.
3. Contact and spatial-relation graphs for each moment.
4. Events defined by changes in those graphs.
5. Preconditions, postconditions, and object-state transitions.
6. Natural-language descriptions for broad semantic grounding.
7. Robot demonstrations connecting abstract physical roles to controls.

The model can then learn several related mappings:

- pixels to entities, resources, contacts, and events;
- ontology history to the next likely event;
- current state and goal to required physical relationships;
- abstract contact plans to embodiment-specific motion;
- actions to predicted next physical states;
- observed failures to corrective actions.

The structured ontology is therefore an interpretable semantic and causal
interface, not necessarily the world model's only internal representation.

## What the ontology does not solve by itself

Semantic task understanding does not provide all information required for
physical execution. A robot also needs:

- three-dimensional position and orientation;
- trajectories, velocities, and timing;
- object geometry and affordances;
- mass, rigidity, friction, and fragility estimates;
- contact forces and grasp stability;
- collision avoidance and reachability;
- embodiment-specific kinematics and dynamics;
- closed-loop control under uncertainty;
- failure detection and recovery behavior.

For example, "fingers hold egg" identifies a relevant relationship but does
not specify grasp aperture, approach trajectory, wrist pose, contact force, or
how to avoid crushing and dropping the egg.

If ontology labels replace the raw spatial and motion signal, performance may
decline because the abstraction discards necessary control information. The
benefit comes from joint training, where structured labels guide attention and
reasoning while the original observations preserve physical detail.

## One-shot task execution

This representation can improve one-shot transfer when a task is composed of
familiar contacts, affordances, and state transitions. A robot that has
learned reusable concepts such as support, stabilize, rotate, insert, and
release can apply those concepts in new combinations.

However, semantic understanding alone does not guarantee perfect one-shot
execution. A more realistic target is:

> Extract an embodiment-independent task model from a demonstration, map its
> physical roles and constraints onto the robot, then execute with feedback
> and online correction.

For tasks covered by the training distribution, an accurate
ontology-enhanced system should be more reliable and data-efficient than an
otherwise equivalent system trained only with general descriptions. The
remaining performance ceiling will be determined primarily by perception,
geometry, dynamics, planning, and closed-loop control.

## Hardware limitations of the current experiments

The results in this project are constrained by the available hardware: two
NVIDIA GTX 1080 Ti GPUs based on the Pascal architecture, with approximately
11 GiB of VRAM each. Pascal remains useful for correctness testing and older
FP16 workloads, but it predates the hardware and software paths used by
current vision-language models.

Important limitations include:

- no BF16 tensor-core acceleration;
- no native FP8 support;
- no modern Tensor Cores for the mixed-precision formats used by current
  model-serving and training stacks;
- limited compatibility with optimized attention and linear-attention
  kernels;
- only 22 GiB aggregate VRAM, split across two devices rather than available
  as one contiguous memory pool;
- high communication and CPU-offload costs when a model cannot fit entirely
  in GPU memory.

Pre-Volta attention also requires special handling. Some scaled-dot-product
attention paths perform unstable FP16 softmax operations on Pascal and can
produce NaNs or corrupt output. The project therefore selects eager attention
for these GPUs, accepting lower throughput in exchange for numerical
stability.

### Constraints even on the 4B model

The Qwen3-VL-4B model is small by current standards, but inference and
fine-tuning still operate close enough to the Pascal limits to require:

- batch size one;
- gradient accumulation instead of larger physical batches;
- gradient checkpointing;
- capped image resolution and vision-token counts;
- eager attention on Pascal;
- model distribution across both GPUs;
- allocator settings that reduce fragmentation from differently sized image
  tensors.

A representative 4B LoRA run over 197 examples for 20 epochs took roughly
12.5 hours on the two GTX 1080 Ti GPUs. This is sufficient to demonstrate that
ontology conventions can be learned, but it makes broad hyperparameter
searches, multiple random seeds, ablation studies, and repeated held-out
evaluations expensive. It also encourages small datasets and low-resolution
frames, which can hide fine contact details that the ontology is intended to
capture.

Consequently, results from the 4B experiments should not be interpreted as
the ceiling of the ontology approach. They are evidence that the pipeline
works under constrained conditions, not a definitive test of how accurately
a well-resourced model can learn contact-oriented representations.

### Qwen3.8-27B trial

Qwen3.8-27B-FP8 could not run natively in FP8 because Pascal has compute
capability 6.1, while the model's FP8 path requires substantially newer GPU
hardware. Transformers had to dequantize its weights to BF16 and place much
of the model in system RAM. A successful configuration used only part of each
GPU's memory and offloaded 53 mapped modules to the CPU so temporary
dequantization and inference tensors would still fit.

That configuration required approximately 13 hours 47 minutes to process 17
frames, averaging about 48.6 minutes per frame. The experiment established
functional compatibility and produced coherent output, but its speed makes
fine-tuning, systematic prompt experiments, and statistically meaningful
evaluation impractical on this machine. A first attempt that allowed the
automatic device map to fill the GPUs more aggressively also failed during
FP8-to-BF16 conversion because insufficient temporary VRAM remained.

### What better hardware would enable

Modern GPUs with native BF16/FP8 Tensor Cores and substantially more VRAM
would allow the ontology theory to be tested rather than mostly testing
memory-management compromises. Suitable hardware would enable:

- native inference of stronger 8B, 27B, and larger vision-language models;
- QLoRA or LoRA fine-tuning of models with enough capacity to perceive small
  contacts and follow a detailed ontology simultaneously;
- higher-resolution inputs and more temporal context;
- larger and more diverse physical batches;
- efficient sweeps over learning rates, ontology formulations, context
  windows, and resource granularity;
- multiple training seeds and proper train/validation/test splits;
- evaluation across genuinely held-out tasks, objects, actors, viewpoints,
  and environments;
- comparison of flat labels, contact graphs, state-transition supervision,
  and combined objectives under otherwise equal conditions;
- enough throughput to annotate and evaluate long videos rather than sparse
  frame subsets.

The central experimental question is therefore still open at scale: how much
does explicit entity-resource-contact-event supervision improve perception,
world modeling, embodiment transfer, and downstream robot execution when both
the ontology-enhanced system and its baseline are trained properly on modern
hardware? The current Pascal experiments support the feasibility of the
representation, but better hardware is required for a fair test of its full
performance.

## Practical lessons from the video experiments

- Strong vision-language models perceive many relevant details but do not
  naturally express them using a stable ontology.
- Fine-tuning is important for consistent resource granularity, persistent
  identities, contact roles, and entity-to-resource transitions.
- Differences in action vocabulary should not automatically be treated as
  failures when the underlying physical meaning is correct.
- Repeated descriptions are valid when the visible state is genuinely
  unchanged; temporal consistency should not be confused with stagnation.
- Intent can be relevant when it is visibly supported by posture and context,
  but observation and inferred intent should remain distinguishable fields.
- Evaluation must measure semantic and physical correctness, not only exact
  agreement with one annotation vocabulary.
- Held objects should transition into resources when they become controlled
  participants in the action.
- Actions should not reference resources absent from the resource/contact
  graph.
- Held-out videos are required to distinguish generalization from memorizing
  training annotations.

## Overall conclusion

Contact-oriented ontology supervision is a promising way to make
demonstration video more useful for general-purpose humanoid learning. Under
otherwise equal conditions, it should improve sample efficiency, task
understanding, embodiment transfer, compositionality, and execution
reliability for tasks represented in training.

Its proper role is to expose the physical and causal structure hidden by
general captions while remaining grounded in raw video, geometry, dynamics,
and feedback. It is a bridge from observation to planning and control, not a
substitute for them.
