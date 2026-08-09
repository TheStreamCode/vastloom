# Contributing

This repository is private during the initial research phase.

Every change must:

1. preserve exact model semantics unless an explicitly named quality-bounded mode is
   being tested;
2. add or update correctness and safety tests;
3. compile with `STARGATE_WARNINGS_AS_ERRORS=ON`;
4. pass ASan and UBSan on Linux;
5. report end-to-end accepted-token results, not only kernel microbenchmarks;
6. include NVMe bytes/token and cache state for memory-placement claims;
7. document negative results and hardware details.

Do not add unbounded queues, implicit OS swap, writes to original weights or automatic
remote telemetry.
