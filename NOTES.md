# Agatha notes

## Proposed world-model architecture

A better architecture could improve collisions and stacks by representing interactions between objects explicitly. This is a proposal, not an implemented or evaluated replacement.

The current model is a per-object MLP with three neighboring objects concatenated into fixed input slots. The proposed replacement is a small graph network that learns contact impulses:

1. **Objects become nodes:** position, orientation, velocity, dimensions, mass, and friction.
2. **Nearby pairs become edges:** relative velocity, surface separation, and relative orientation. The support table participates too.
3. **Messages propagate through the graph:** several message-passing rounds allow a push into a lower book to influence the books above it.
4. **The network predicts interaction impulses:** these update velocities, then a numerical integrator advances positions and orientations. Equal-and-opposite pairwise impulses can enforce conservation of linear momentum for internal interactions. Gravity and support interactions must be handled as external influences on the movable objects. Conserving angular momentum requires additional structure; equal-and-opposite linear impulses alone do not guarantee it.

### Expected benefits and evaluation

The hypothesis is better handling of stacks, object ordering, and different object counts. Improvement is not guaranteed and must be measured against the MLP on identical data and held-out scenes. Graph simulators have demonstrated learned rigid-contact dynamics, but those results do not establish performance for Agatha's scenes.

Measure:

- Mean position error at 0.5, 1, and 2 seconds.
- Table penetration frequency and depth.
- Fall precision and recall, alongside overall fall/no-fall agreement.
- Rollout stability and inference time.

### Training changes

The current model trains on one-step targets but runs autoregressively at inference. Training over short autoregressive sequences could reduce the mismatch between training inputs and the model's own imperfect predictions. Noise during training is another established approach to improving rollout robustness.

For Agatha, the proposed first experiment is **2–3 message-passing rounds plus multi-step training**, keeping the existing MLP as the baseline. Retain a replacement only after evaluation supports it.

### Novelty and scope

Graph networks and momentum-conserving impulse architectures already exist. Describe this as our implementation and adaptation; original research novelty is unproven. This proposal stays within the existing object-state world-model scope and does not require new product features.

### References

- [Graph network simulators can learn discontinuous, rigid contact dynamics](https://proceedings.mlr.press/v205/allen23a.html).
- [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405).
- [Dynami-CAL GraphNet: A Physics-Informed Graph Neural Network Conserving Linear and Angular Momentum for Dynamical Systems](https://arxiv.org/abs/2501.07373).
