# Local ret2win fixture

This is an intentionally vulnerable local-only pwn fixture.  Connect to the
supplied TCP target and obtain its runtime-random flag.  The handout contains
the exact source used to compile the service binary so deterministic tests do
not depend on a checked-in architecture-specific ELF.

