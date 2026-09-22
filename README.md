# Andrew Smith

### AI Security & Assurance · LLM & Agent Security · Formal Assurance · Security Architecture

I am a cybersecurity and applied-research practitioner exploring how **formal reasoning, security architecture and machine-verifiable properties** can provide stronger assurance for AI and other digital systems.

My background combines enterprise cybersecurity and technology experience with independent research into LLM/agent security, access control, cryptographic authorisation and software assurance.

> **Development approach:** My implementation work is substantially LLM-assisted. I originate the research questions and system concepts, define requirements and architecture, direct the technical development and experiments, and evaluate and critique the resulting systems. The projects below are research prototypes, not production systems.

## Current focus

- **AI security & assurance** — prompt injection, instruction integrity, agent/tool boundaries and defence-in-depth
- **Identity & access** — default-deny policy design, authorisation and formally specified security properties
- **Formal assurance** — Coq proof artefacts and machine-verifiable access-control properties
- **Cryptographic authorisation** — signed proofs, protocol design and secure communications
- **Software assurance** — static analysis, constrained execution and behavioural validation

## Selected research prototypes

| Project | Research question | Evidence / status |
|---|---|---|
| **[Secure-LLM-Gateway](portfolio/secure-llm-gateway.md)** | Can access-control and consistency mechanisms strengthen defence-in-depth for LLM and agent systems? | Private research prototype; policy enforcement, tool controls, session consistency and integrity mechanisms |
| **[FVAC](portfolio/fvac.md)** | Can formally verifiable access-control properties complement conventional identity systems? | Private research prototype; Python + Coq formalisation with completed proof scripts for specific access-control properties |
| **[Proof-Messenger](portfolio/proof-messenger.md)** | Can cryptographic authorisation improve assurance in communications systems? | Rust/WASM research prototype; Ed25519 and unit/integration/property-based tests; [public demos](https://github.com/DeepFriedCyber/proof-messenger-demos) |
| **[VeriPy](portfolio/veripy.md)** | Can verifiable software properties help identify inconsistent modification or behaviour? | Private Python research prototype; AST analysis, constrained execution and behavioural validation |

## How I work

A recurring pattern across my research is:

**Problem observed → hypothesis → research → security/mathematical model → LLM-assisted prototype → testing and evaluation → limitations identified**

I use modern LLMs as engineering tools to make research ideas executable and testable. I do not present this as conventional hand-coded software-engineering experience.

## Background

Before my current research work, I spent much of my career around enterprise technology and cybersecurity, including roles with **VeriSign**, **PGP**, **Entrust Datacard**, **CA Technologies** and **Quest Software / Vizioncore**. I also founded **Deep Fried Cyber**, a B2B cybersecurity marketplace that grew to more than 2,500 listed companies.

## Interested in

I am particularly interested in opportunities involving **AI Security, AI Assurance, LLM/Agent Security, Security Architecture, Security Research and emerging-technology consulting**.

[LinkedIn](https://www.linkedin.com/in/andrewsmith46/)

---

*All projects described here are research prototypes. Claims are intentionally limited to properties demonstrated by the relevant implementation, tests or formal artefacts.*
