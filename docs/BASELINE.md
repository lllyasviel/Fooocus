# Fooocus Modern — Upstream Baseline

## Project

Fooocus Modern is a modern image-generation platform derived from Fooocus.

## Upstream Project

Repository:
https://github.com/lllyasviel/Fooocus

## Local Upstream Baseline

Commit:
ae05379

Branch:
main

## Fork

Repository:
https://github.com/WURAlex/Fooocus-Modern

## Development Branch

modern-platform

## Baseline Branch

upstream-baseline

## Purpose

This document records the upstream source state from which Fooocus
Modern development begins.

The upstream Fooocus implementation is initially retained as the
functional reference implementation. Future development will
progressively modularize the application and introduce support for
modern image-generation architectures while preserving existing
Fooocus functionality.

## Initial Objectives

- Maintain existing Fooocus functionality.
- Support current Google Colab environments.
- Establish reproducible dependency management.
- Remove uncontrolled runtime self-updating.
- Improve model download and integrity verification.
- Introduce hardware-aware execution.
- Introduce modular generation engines.
- Support modern architectures such as FLUX.
- Maintain a secure and maintainable codebase.

## Important Principle

Fooocus Modern is a single long-term project.

A separate "Option A" fork will not be maintained.

The existing Fooocus functionality is the initial compatibility
baseline for the modern platform.