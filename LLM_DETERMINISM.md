# LLM Determinism Implementation

## Overview

This document describes the implementation of deterministic behavior in the `prompt_market.py` script. The goal is to ensure consistent trading recommendations when running the LLM with the same input.

## Changes Made

The following changes were made to enhance determinism:

1. **Enforced deterministic parameters in `get_trading_recommendation`:**
   - Set `temperature = 0` 
   - Set `top_p = 0`
   - Set fixed `seed = 42`

2. **Added model compatibility checks:**
   - Created a dictionary of models that don't support deterministic parameters
   - Implemented fallback to default values for unsupported parameters
   - Added warning messages when using models without full deterministic support

3. **Added deterministic parameters to external API calls:**
   - Updated `get_hyperbolic_response` to accept and use deterministic parameters
   - Updated `get_ollama_response` to accept and use deterministic parameters

4. **Added documentation about deterministic behavior**:
   - Added docstring to functions explaining deterministic limitations
   - Created testing scripts to verify deterministic settings

## Model Compatibility

Some models do not support certain deterministic parameters:

| Model    | Unsupported Parameters |
|----------|------------------------|
| o4-mini  | temperature, top_p     |

For these models, the code skips setting the unsupported parameters and falls back to using default values. This ensures that the model still works, even if it can't provide the same level of determinism as other models.

## Testing Results

As verified by the `verify_prompt_market_determinism.py` script, all deterministic parameters are properly set in the code. However, our previous testing with `verify_deterministic_llm.py` showed some important findings:

### Main Findings

1. **Hash-level Determinism**: Even with `temperature=0`, `top_p=0`, and a fixed `seed`, OpenAI models (e.g., gpt-4o and gpt-4o-mini) do not produce byte-identical outputs across multiple runs with the same prompt.

2. **Critical Field Determinism**: For trading recommendations, the critical decision fields (signal type, prices, stop loss) generally remain consistent, with variations primarily in:
   - Wording in the `REASONING` field
   - Minor numerical variations in calculated fields like `R/R_RATIO` (e.g., 2.14 vs 2.17)

3. **Recommended Usage**: For applications requiring absolute determinism at the byte level, these parameters are insufficient. However, for trading signals where functional equivalence is more important than byte-level reproducibility, these settings provide reasonable consistency in the critical decision fields.

## Verification

We created two verification scripts:

1. `verify_deterministic_llm.py` - Runs actual tests against LLMs and compares outputs
2. `verify_prompt_market_determinism.py` - Verifies that deterministic parameters are properly set in the code

## Conclusion

While absolute byte-level determinism cannot be guaranteed with current LLM APIs, the implemented changes provide the best possible consistency for trading recommendations. Critical trading decision fields (signal type, prices, stop loss) should remain mostly consistent across runs, though some variation in text fields and minor numerical calculations may still occur.

Models like o4-mini that don't support certain deterministic parameters will show even more variation in outputs, but they can still be used with the script.

## Further Improvements

If stronger determinism is required:

1. **Post-processing**: Implement post-processing to normalize or standardize the output format
2. **Extraction and normalization**: Extract only the critical fields and normalize their values
3. **Result caching**: Cache results for identical inputs to ensure consistent responses
4. **Model selection**: Use only models that fully support deterministic parameters 