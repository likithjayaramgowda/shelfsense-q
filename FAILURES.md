# Failure atlas

Every compile, quantize, export, or profile failure — what broke, the error
code, what changed, whether it worked.

This is the highest-value-per-hour artifact in the project. It costs nothing
but honesty, and it is the section of the README that separates someone who
ran a tutorial from someone who debugged a deployment.

Template:

```
## YYYY-MM-DD — short title

**Context:** what I was trying to do
**Error:** exact message + code
**Hypothesis:** what I thought was wrong
**Fix:** what I actually changed
**Outcome:** worked / didn't / partially
**Lesson:** the transferable bit
```

Useful reference — common QAIRT/QNN error meanings:

- **3110** — input/output types not supported by the QNN backend. Usually an
  unquantized tensor passed through a quantized graph. Fix: quantize or
  re-quantize the model.
- **Unsupported rank** — most ops are <=5D; some do 5D; none do >5D.
- **Unsupported type** — common with int32 layers.

---

*(no entries yet)*
