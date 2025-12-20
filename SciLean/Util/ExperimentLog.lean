/--
Helpers for appending benchmark/experiment entries to a markdown log.
-/
import Std

namespace SciLean

namespace ExperimentLog

/-- Structured log entry for experiments. -/
structure Entry where
  timestamp : String
  title : String := ""
  commit : Option String := none
  command : Option String := none
  notes : List String := []
  body : List String := []

/-- Render a log entry to markdown. -/
def Entry.render (e : Entry) : String :=
  let header := s!"## {e.timestamp}\n\n"
  let title := if e.title.isEmpty then "" else s!"### {e.title}\n\n"
  let commit := match e.commit with
    | some c => s!"Commit: {c}\n\n"
    | none => ""
  let command := match e.command with
    | some c => s!"Command: `{c}`\n\n"
    | none => ""
  let notes :=
    if e.notes.isEmpty then ""
    else
      let lines := e.notes.map (fun n => s!"- {n}\n") |> String.join
      s!"Notes:\n{lines}\n"
  let body :=
    if e.body.isEmpty then ""
    else (e.body.map (fun l => s!"{l}\n") |> String.join) ++ "\n"
  header ++ title ++ commit ++ command ++ notes ++ body

/-- Append text to a file, creating it if needed. -/
def appendFile (path : System.FilePath) (content : String) : IO Unit := do
  if (← path.pathExists) then
    let existing ← IO.FS.readFile path
    IO.FS.writeFile path (existing ++ content)
  else
    IO.FS.writeFile path content

/-- Ensure the markdown log has a top-level header. -/
def ensureHeader (path : System.FilePath) : IO Unit := do
  unless (← path.pathExists) do
    IO.FS.writeFile path "# Experiment Log\n\n"

/-- Append an {name}`Entry` to the markdown log at {lit}`path`. -/
def append (path : System.FilePath) (entry : Entry) : IO Unit := do
  ensureHeader path
  appendFile path (entry.render)

end ExperimentLog

end SciLean
