# Leakage Analysis

## Summary

The repository’s leakage audit script ran successfully and reported zero duplicate checksums across splits. However, the current split protocol is machine-dependent and should be considered a leakage risk for machine-independent generalization claims.

## Evidence

- The leakage audit script executed successfully and reported the current audit as PASSED.
- The generated manifest shows that the same machine IDs appear in both train and validation splits.
- The protocol therefore does not satisfy a strict machine-independent evaluation definition.

## Recommendation

If the goal is to claim generalization to unseen machine IDs, the split must be changed so that machine IDs are disjoint across train, validation, and test sets.
