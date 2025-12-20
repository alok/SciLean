import Std

/-!
Minimal HTTP helper using {lit}`curl` for TLS, built around raw request data.
-/

namespace Wandb

/-- Raw HTTP response from {lit}`curl`. -/
structure Response where
  exitCode : Int
  stdout : String
  stderr : String

/-- Raw HTTP request specification. -/
structure Request where
  method : String := "POST"
  url : String
  headers : List (String × String) := []
  body : Option String := none
  timeoutMs : Option Nat := none

/-- Build {lit}`curl` arguments from a request. -/
def Request.toCurlArgs (r : Request) : Array String :=
  let base := ["-sS", "-X", r.method, "--fail"]
  let headerArgs :=
    r.headers.foldl (init := []) fun acc (k, v) =>
      acc ++ ["-H", s!"{k}: {v}"]
  let timeoutArgs :=
    match r.timeoutMs with
    | none => []
    | some t =>
      let secs := (Float.ofNat t) / 1000.0
      ["--max-time", toString secs]
  let bodyArgs := match r.body with
    | none => []
    | some body => ["--data", body]
  (base ++ headerArgs ++ timeoutArgs ++ bodyArgs ++ [r.url]).toArray

/-- Execute a raw HTTP request via {lit}`curl`. -/
def run (r : Request) : IO Response := do
  let out ← IO.Process.output {
    cmd := "curl"
    args := r.toCurlArgs
  }
  let exitCode := Int.ofNat out.exitCode.toNat
  pure { exitCode := exitCode, stdout := out.stdout, stderr := out.stderr }

/-- Convenience helper for posting JSON. -/
def postJson (url : String) (headers : List (String × String)) (body : String) : IO Response :=
  run {
    method := "POST"
    url := url
    headers := ("Content-Type", "application/json") :: headers
    body := some body
  }

end Wandb
