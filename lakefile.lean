import Lake
open Lake DSL

package optlib where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩, -- pretty-prints `fun a ↦ b`
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩]

@[default_target]
lean_lib Optlib where

require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "v4.18.0"

meta if get_config? env = some "CI_BUILD" then
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4.git" @ "v4.18.0"
