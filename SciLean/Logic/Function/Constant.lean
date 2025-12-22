
/-- {name}`Function.IsConstant` for {given}`f` states that {lean}`f` is a constant function. -/
def Function.IsConstant (f : α → β) : Prop := ∀ x y, f x = f y
