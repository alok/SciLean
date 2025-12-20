import Std
import Wandb.Http

/-! Tiny W&B client helpers built on top of the raw HTTP layer. -/

namespace Wandb

/-- W&B client configuration. -/
structure Config where
  baseUrl : String := "https://api.wandb.ai"
  apiKey : String
  entity : Option String := none
  project : Option String := none

/-- Read {lit}`WANDB_API_KEY` from the environment. -/
def apiKeyFromEnv : IO (Option String) :=
  IO.getEnv "WANDB_API_KEY"

/-- Build a config from environment variables. -/
def Config.fromEnv : IO Config := do
  let some key ← apiKeyFromEnv
    | throw <| IO.userError "WANDB_API_KEY not set"
  let entity ← IO.getEnv "WANDB_ENTITY"
  let project ← IO.getEnv "WANDB_PROJECT"
  pure { apiKey := key, entity := entity, project := project }

/-- Default auth header for W&B. -/
def authHeader (apiKey : String) : (String × String) :=
  ("Authorization", s!"Bearer {apiKey}")

/-- Minimal JSON string escape. -/
def escapeJson (s : String) : String :=
  s.foldl (init := "") fun acc c =>
    match c with
    | '"' => acc ++ "\\\""
    | '\\' => acc ++ "\\\\"
    | '\n' => acc ++ "\\n"
    | '\r' => acc ++ "\\r"
    | '\t' => acc ++ "\\t"
    | _ => acc.push c

/-- Build a JSON object with {lit}`query` and optional {lit}`variables`. -/
def graphqlBody (query : String) (variables : Option String) : String :=
  let q := escapeJson query
  match variables with
  | none => "{\"query\":\"" ++ q ++ "\"}"
  | some vars =>
    let v := escapeJson vars
    "{\"query\":\"" ++ q ++ "\",\"variables\":\"" ++ v ++ "\"}"

/-- Post a GraphQL query to W&B's endpoint. -/
def postGraphQL (cfg : Config) (query : String) (variables : Option String := none) : IO Response := do
  let url := cfg.baseUrl ++ "/graphql"
  postJson url [authHeader cfg.apiKey] (graphqlBody query variables)

/-- Post a raw JSON payload to a W&B endpoint path. -/
def postJsonPath (cfg : Config) (path : String) (body : String) : IO Response := do
  let url := cfg.baseUrl ++ path
  postJson url [authHeader cfg.apiKey] body

end Wandb
