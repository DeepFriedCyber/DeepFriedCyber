# Proof-Messenger

**Status:** Research prototype  
**Area:** Cryptographic authorisation / secure communications

## Research question

Can stronger cryptographic authorisation mechanisms improve the assurance available in messaging and communications systems?

## What I explored

Proof-Messenger is a multi-component Rust/WASM prototype containing protocol, relay, CLI and web components.

Repository evidence includes:

- Ed25519 signing and verification
- proof creation and validation
- incorrect-key rejection tests
- modified-content rejection tests
- unit, integration and property-based testing

Post-quantum cryptography is a research/roadmap direction rather than a claim about the current core protocol.

## My role

I originated the system concept and security requirements and directed LLM-assisted development, testing and iterative refinement.

A public demonstration repository is available at [proof-messenger-demos](https://github.com/DeepFriedCyber/proof-messenger-demos).
