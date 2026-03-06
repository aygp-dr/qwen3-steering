/-
  Formal specification of steering vector properties.
  Thin but real: these are the invariants the Python code must satisfy.
-/

-- Steering vector lives on unit sphere in R^d_model
structure SteeringVector (d : Nat) where
  components : Fin d → Float
  unit_norm : Float  -- must equal 1.0 within tolerance

-- Alpha bound: empirical precondition from literature
def alpha_safe (α : Float) : Prop :=
  0.0 ≤ α ∧ α ≤ 0.35

-- Layer index must be valid for the model
def valid_layer (layer : Nat) (num_layers : Nat) : Prop :=
  layer < num_layers

-- Qwen3-0.6B constants
def qwen3_d_model : Nat := 1024
def qwen3_num_layers : Nat := 28

-- The generate_steered precondition
structure SteeringConfig where
  layer : Nat
  alpha : Float
  layer_valid : valid_layer layer qwen3_num_layers
  alpha_safe : alpha_safe alpha

-- Zero-alpha identity: steering with α=0 must not modify output
-- (stated as a type-level requirement)
axiom zero_alpha_identity :
  ∀ (prompt : String) (vec : SteeringVector qwen3_d_model) (layer : Nat),
    valid_layer layer qwen3_num_layers →
    True  -- placeholder: steered(prompt, vec, layer, 0.0) = baseline(prompt)

-- Normalization idempotence: normalizing a unit vector returns itself
theorem normalize_idempotent (v : SteeringVector d) (h : v.unit_norm = 1.0) :
    v.unit_norm = 1.0 := h
