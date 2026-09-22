# Secure-LLM-Gateway

**Status:** Research prototype  
**Area:** LLM & agent security / AI assurance

## Research question

Can access-control, integrity and structural-consistency mechanisms contribute to defence-in-depth for LLM and agent systems?

## What I explored

Secure-LLM-Gateway grew from earlier work on FVAC and an investigation into whether access-control ideas could be applied to prompt-injection and instruction-integrity risks.

The prototype explores mechanisms including:

- default-deny policy enforcement
- authentication-level and asset-criticality controls
- tool allowlisting
- session consistency and state-aware controls
- Merkle-based integrity mechanisms
- sheaf-inspired consistency structures
- security-event and audit mechanisms

## My role

I originated the research direction, requirements and architecture and directed the iterative development and evaluation of the prototype using LLM-assisted engineering.

## Important limitation

This is an experimental defence-in-depth architecture. It **does not claim that prompt injection can be universally or mathematically prevented**.

The implementation repository is currently private.
