# FVAC — Formally Verified Access Control

**Status:** Research prototype  
**Area:** Formal assurance / access control / identity security

## Research question

Can formally verifiable access-control properties provide a lightweight complementary assurance layer for identity and authorisation systems?

## What I explored

FVAC contains a Python application/prototype layer and a Coq formalisation of core access-control behaviour. Repository artefacts include completed Coq proof scripts covering properties such as:

- no-read-up behaviour
- read-down behaviour
- access-level ordering
- inactive-user denial

The wider prototype also includes testing and post-quantum cryptography integration experiments.

## My role

I originated the security concept and research direction, specified the intended access-control behaviour and directed LLM-assisted development of the software and formal artefacts.

## Important limitation

The Coq artefacts formalise specific access-control properties. They should not be interpreted as a proof that the complete software system is secure or quantum-secure.

The implementation repository is currently private.
