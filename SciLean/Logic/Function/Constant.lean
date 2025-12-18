
/-- {name}`Function.IsConstant` {given}`f` implies that function {given}`f` is a constant function. -/
def Function.IsConstant (f : α → β) : Prop := ∀ x y, f x = f y
