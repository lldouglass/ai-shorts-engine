"""Hook formulas for short-form video content.

Research shows 65% of viewers who make it past 3 seconds watch to 10 seconds,
and 45% of those stick to 30 seconds. The hook is EVERYTHING.

These formulas are injected into story/topic generation prompts.
"""

HOOK_FORMULAS = """## Proven Hook Formulas (use one per video)

### Curiosity Gap Hooks (highest watch-through rate):
- "The [thing] nobody talks about with [subject]..."
- "[Number] [things] that [surprising outcome]"
- "I checked the data on [X] and found something weird"
- "This [car/stock] has a secret most people don't know"

### Fear/Protection Hooks (high save rate):
- "NEVER [do this thing] with your [subject]"
- "Stop buying [X] before you see this"
- "[X]% of people get this wrong about [subject]"

### Data/Authority Hooks (high share rate):
- "I analyzed [large number] [things] and here's what I found"
- "The data says [counterintuitive finding]"
- "According to [credible source], [surprising stat]"

### Challenge/Poll Hooks (highest comment rate):
- "Which would you pick: A or B?"
- "Rate this [0-10] - most people get it wrong"
- "Can you guess which [thing] [outcome]?"

### Rules for ALL Hooks:
- Must work as TEXT OVERLAY (silent viewers)
- Max 10 words on screen
- First visual frame must be striking/unexpected
- Never start with a logo, intro, or greeting
- Get to the point in under 2 seconds"""
