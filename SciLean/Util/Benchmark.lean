import Lean
import Wandb.Json
import Wandb.Local

/-!
# Benchmarking Utilities

Reusable benchmarking infrastructure for SciLean.

Benchmark results can optionally be logged to a local W&B run. Disable with
{lit}`SCILEAN_WANDB_BENCH=0`, {lit}`WANDB_DISABLED=true`, or
{lit}`WANDB_MODE=disabled`. Local runs are written under {lit}`.wandb` or
{lit}`wandb`, and can be synced with {lit}`wandb sync`.
-/

namespace SciLean.Benchmark

/-- Repeat a character n times -/
def padRight (s : String) (width : Nat) (c : Char := ' ') : String :=
  let padding := if s.length < width then String.ofList (List.replicate (width - s.length) c) else ""
  s ++ padding

/-- Configuration for a benchmark run -/
structure Config where
  warmupIterations : Nat := 3
  timedIterations : Nat := 10
  printProgress : Bool := false
  deriving Repr

/-- Result of a single benchmark -/
structure Result where
  name : String
  avgTimeNs : Nat
  minTimeNs : Nat
  maxTimeNs : Nat
  iterations : Nat
  deriving Repr, Inhabited

namespace WandbBench

def envBool? (name : String) : IO (Option Bool) := do
  match (← IO.getEnv name) with
  | none => pure none
  | some raw =>
      let v := raw.trimAscii.toString.toLower
      if v == "1" || v == "true" || v == "yes" || v == "on" then
        pure (some true)
      else if v == "0" || v == "false" || v == "no" || v == "off" then
        pure (some false)
      else
        pure none

def computeEnabled : IO Bool := do
  match (← envBool? "SCILEAN_WANDB_BENCH") with
  | some v => pure v
  | none =>
      match (← envBool? "WANDB_DISABLED") with
      | some true => pure false
      | _ =>
          match (← IO.getEnv "WANDB_MODE").map String.toLower with
          | some "disabled" | some "dryrun" => pure false
          | _ => pure true

initialize enabledRef : IO.Ref (Option Bool) ← IO.mkRef none
initialize runRef : IO.Ref (Option _root_.Wandb.Local.LocalRun) ← IO.mkRef none

def platformName : String :=
  if System.Platform.isOSX then
    "macos"
  else if System.Platform.isWindows then
    "windows"
  else
    "linux"

def optField (key : String) (value? : Option String) : List (String × _root_.Wandb.Json.J) :=
  match value? with
  | some v => [(key, _root_.Wandb.Json.str v)]
  | none => []

def initRun : IO _root_.Wandb.Local.LocalRun := do
  let run ← _root_.Wandb.Local.init
  let program ← IO.appPath
  let leanVersion := Lean.versionString
  let leanGithash := Lean.githash
  let toolchain ← IO.getEnv "LEAN_TOOLCHAIN"
  let sysroot ← IO.getEnv "LEAN_SYSROOT"
  let leanPath ← IO.getEnv "LEAN_PATH"
  let lake ← IO.getEnv "LAKE"
  let lakeEnv ← IO.getEnv "LAKE_ENV"
  let scileanCommit ← IO.getEnv "SCILEAN_GIT_COMMIT"
  let scileanBranch ← IO.getEnv "SCILEAN_GIT_BRANCH"
  let baseFields : List (String × _root_.Wandb.Json.J) := [
    ("program", _root_.Wandb.Json.str program.toString),
    ("os", _root_.Wandb.Json.str platformName),
    ("scilean_benchmark", _root_.Wandb.Json.bool true),
    ("lean_version", _root_.Wandb.Json.str leanVersion),
    ("lean_githash", _root_.Wandb.Json.str leanGithash)
  ]
  let metadataFields :=
    baseFields
    ++ optField "lean_toolchain" toolchain
    ++ optField "lean_sysroot" sysroot
    ++ optField "lean_path" leanPath
    ++ optField "lake" lake
    ++ optField "lake_env" lakeEnv
    ++ optField "scilean_git_commit" scileanCommit
    ++ optField "scilean_git_branch" scileanBranch
  let metadata := _root_.Wandb.Json.obj metadataFields
  _root_.Wandb.Local.writeMetadata run.paths metadata
  _root_.Wandb.Local.writeConfig run.paths [("scilean_benchmark", _root_.Wandb.Json.bool true)]
  _root_.Wandb.Local.writeSummary run.paths (_root_.Wandb.Json.obj [])
  pure run

def isEnabled : IO Bool := do
  match (← enabledRef.get) with
  | some v => pure v
  | none =>
      let v ← computeEnabled
      enabledRef.set (some v)
      pure v

def getRun : IO (Option _root_.Wandb.Local.LocalRun) := do
  if !(← isEnabled) then
    return none
  match (← runRef.get) with
  | some run => pure (some run)
  | none =>
      let run ← initRun
      runRef.set (some run)
      pure (some run)

def logFields (fields : List (String × _root_.Wandb.Json.J)) : IO Unit := do
  try
    let some run ← getRun | return ()
    let run ← _root_.Wandb.Local.log run fields
    runRef.set (some run)
  catch _ =>
    pure ()

def logResult (r : Result) (config : Config) : IO Unit := do
  let nsToS (ns : Nat) : Float :=
    ns.toFloat / 1_000_000_000.0
  let fields : List (String × _root_.Wandb.Json.J) := [
    ("benchmark/name", _root_.Wandb.Json.str r.name),
    ("benchmark/avg_time_ns", _root_.Wandb.Json.nat r.avgTimeNs),
    ("benchmark/min_time_ns", _root_.Wandb.Json.nat r.minTimeNs),
    ("benchmark/max_time_ns", _root_.Wandb.Json.nat r.maxTimeNs),
    ("benchmark/avg_time_s", _root_.Wandb.Json.float (nsToS r.avgTimeNs)),
    ("benchmark/min_time_s", _root_.Wandb.Json.float (nsToS r.minTimeNs)),
    ("benchmark/max_time_s", _root_.Wandb.Json.float (nsToS r.maxTimeNs)),
    ("benchmark/iterations", _root_.Wandb.Json.nat r.iterations),
    ("benchmark/warmup_iterations", _root_.Wandb.Json.nat config.warmupIterations),
    ("benchmark/timed_iterations", _root_.Wandb.Json.nat config.timedIterations)
  ]
  logFields fields

end WandbBench

/-- Log a scalar benchmark metric to local W&B, if enabled. -/
def logScalar (name : String) (value : Float) (unit? : Option String := none) : IO Unit := do
  let base := [
    ("benchmark/name", _root_.Wandb.Json.str name),
    ("benchmark/value", _root_.Wandb.Json.float value)
  ]
  let fields :=
    match unit? with
    | some unit => base ++ [("benchmark/unit", _root_.Wandb.Json.str unit)]
    | none => base
  WandbBench.logFields fields

/-- Log a structured benchmark metric to local W&B, if enabled. -/
def logMetric
    (group : String)
    (metric : String)
    (value : Float)
    (unit? : Option String := none)
    (params : List (String × _root_.Wandb.Json.J) := []) : IO Unit := do
  let base := [
    ("benchmark/group", _root_.Wandb.Json.str group),
    ("benchmark/metric", _root_.Wandb.Json.str metric),
    ("benchmark/value", _root_.Wandb.Json.float value)
  ]
  let fields :=
    match unit? with
    | some unit => base ++ [("benchmark/unit", _root_.Wandb.Json.str unit)]
    | none => base
  let paramFields := params.map (fun (k, v) => (s!"benchmark/param/{k}", v))
  WandbBench.logFields (fields ++ paramFields)

/-- Build a string parameter for {name}`logMetric`. -/
def paramStr (key : String) (value : String) : String × _root_.Wandb.Json.J :=
  (key, _root_.Wandb.Json.str value)

/-- Build a natural number parameter for {name}`logMetric`. -/
def paramNat (key : String) (value : Nat) : String × _root_.Wandb.Json.J :=
  (key, _root_.Wandb.Json.nat value)

/-- Build a float parameter for {name}`logMetric`. -/
def paramFloat (key : String) (value : Float) : String × _root_.Wandb.Json.J :=
  (key, _root_.Wandb.Json.float value)

/-- Format time in appropriate units -/
def formatTime (ns : Nat) : String :=
  if ns >= 1_000_000_000 then
    s!"{ns.toFloat / 1_000_000_000.0}s"
  else if ns >= 1_000_000 then
    s!"{ns.toFloat / 1_000_000.0}ms"
  else if ns >= 1_000 then
    s!"{ns.toFloat / 1_000.0}us"
  else
    s!"{ns}ns"

/-- Pretty print a benchmark result -/
def Result.toString (r : Result) : String :=
  s!"{r.name}: {formatTime r.avgTimeNs} avg ({r.iterations} iterations)"

instance : ToString Result := ⟨Result.toString⟩

/-- Run a benchmark with the given configuration -/
def run [Monad m] [MonadLiftT IO m] (name : String) (config : Config := {})
    (f : Unit → m α) : m Result := do
  -- Warmup
  for _ in [0:config.warmupIterations] do
    let _ ← f ()

  -- Timed runs
  let mut times : Array Nat := #[]
  for i in [0:config.timedIterations] do
    let start ← liftM (m := IO) IO.monoNanosNow
    let _ ← f ()
    let elapsed := (← liftM (m := IO) IO.monoNanosNow) - start
    times := times.push elapsed
    if config.printProgress then
      liftM (m := IO) <| IO.print s!"\r{name}: iteration {i + 1}/{config.timedIterations}"

  if config.printProgress then
    liftM (m := IO) <| IO.println ""

  let sorted := times.toList.mergeSort (· ≤ ·)
  let total := times.foldl (· + ·) 0

  let result := {
    name := name
    avgTimeNs := total / config.timedIterations
    minTimeNs := sorted.head!
    maxTimeNs := sorted.getLast!
    iterations := config.timedIterations
  }
  liftM (m := IO) <| WandbBench.logResult result config
  return result

/-- Run a pure benchmark (forces evaluation) -/
def runPure (name : String) (config : Config := {}) (f : Unit → α) : IO Result := do
  run name config fun () => do
    let _ := f ()
    pure ()

/-- Print a comparison table of results -/
def printComparison (results : Array Result) : IO Unit := do
  IO.println "┌────────────────────────────────────┬────────────────┬──────────────────┐"
  IO.println "│ Benchmark                          │ Avg Time       │ Speedup          │"
  IO.println "├────────────────────────────────────┼────────────────┼──────────────────┤"

  let baseline := (results[0]?).map (·.avgTimeNs) |>.getD 1
  for r in results do
    let speedup := if r.avgTimeNs > 0 then baseline.toFloat / r.avgTimeNs.toFloat else 0
    let nameStr := padRight (r.name.take 34 |>.toString) 34
    let timeStr := padRight (formatTime r.avgTimeNs) 14
    let speedupStr := if speedup >= 1.05 then s!"{Float.round (speedup * 10) / 10}x faster"
                      else if speedup <= 0.95 then s!"{Float.round (10 / speedup) / 10}x slower"
                      else "baseline"
    IO.println s!"│ {nameStr} │ {timeStr} │ {padRight (speedupStr.take 16 |>.toString) 16} │"

  IO.println "└────────────────────────────────────┴────────────────┴──────────────────┘"

/-- Suite of related benchmarks -/
structure Suite where
  name : String
  description : String := ""
  results : Array Result := #[]
  deriving Repr

/-- Add a result to a suite -/
def Suite.add (s : Suite) (r : Result) : Suite :=
  { s with results := s.results.push r }

/-- Print a suite's results -/
def Suite.print (s : Suite) : IO Unit := do
  IO.println s!"\n{s.name}"
  IO.println (String.ofList (List.replicate s.name.length '='))
  if s.description.length > 0 then
    IO.println s.description
  printComparison s.results

end SciLean.Benchmark
