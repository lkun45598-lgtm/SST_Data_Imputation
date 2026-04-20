# Claude Code Guidelines for SST Data Imputation

## Model Usage Strategy

**Optimize Claude model selection for task efficiency**:
- Use **Haiku** for: file reading, parsing, exploration, data collection
- Use **Opus** for: code writing, complex analysis, architecture decisions, debugging
- **Rule**: Always use Haiku for Read/Glob/Grep operations. Switch to Opus before writing code or performing deep analysis.
- **Rationale**: Haiku is fast for I/O tasks; Opus is needed for code quality and complex reasoning.
- **Implementation**: When workflow requires both reading + analysis, do reads in Haiku, then request model switch to Opus for synthesis.
