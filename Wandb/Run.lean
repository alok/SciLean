import Wandb.Json

/-! Run metadata helpers. -/

namespace Wandb

open Wandb.Json

/-- Minimal run reference. -/
structure RunRef where
  entity : String
  project : String
  id : String

/-- Build JSON for a run reference. -/
def RunRef.toJson (r : RunRef) : Json.J :=
  Json.obj
    [ ("entity", Json.str r.entity)
    , ("project", Json.str r.project)
    , ("id", Json.str r.id)
    ]

/-- Build a run reference from environment variables. -/
def RunRef.fromEnv : IO RunRef := do
  let some entity ← IO.getEnv "WANDB_ENTITY"
    | throw <| IO.userError "WANDB_ENTITY not set"
  let some project ← IO.getEnv "WANDB_PROJECT"
    | throw <| IO.userError "WANDB_PROJECT not set"
  let some runId ← IO.getEnv "WANDB_RUN_ID"
    | throw <| IO.userError "WANDB_RUN_ID not set"
  pure { entity := entity, project := project, id := runId }

end Wandb
