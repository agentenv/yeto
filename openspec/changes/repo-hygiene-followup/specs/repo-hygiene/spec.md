## ADDED Requirements

### Requirement: Commit messages contain no Chinese
The subject and body of a new commit SHALL NOT contain CJK Han characters or full-width punctuation. A local `commit-msg` hook SHALL run the check. Past commits SHALL NOT be rewritten.

#### Scenario: Chinese subject is rejected
- **WHEN** a developer commits with Han characters in the subject, and the `commit-msg` hook is installed
- **THEN** the hook exits with a non-zero code and no commit is created

#### Scenario: Full-width punctuation in the body is rejected
- **WHEN** a developer commits with full-width punctuation in the body, and the `commit-msg` hook is installed
- **THEN** the hook exits with a non-zero code and no commit is created

#### Scenario: Non-CJK non-ASCII characters are allowed
- **WHEN** the commit body contains `→` or `é` and no Han characters
- **THEN** the check passes

### Requirement: New ignore rules do not hit tracked files
Each addition to `.gitignore` SHALL ensure that `git ls-files -ci --exclude-standard` shows no new hits (baseline on 2026-10-13: 21).

#### Scenario: Check tracked files after adding a rule
- **WHEN** a maintainer adds a new rule to `.gitignore`
- **THEN** the output of `git ls-files -ci --exclude-standard` is the same as before the rule was added

### Requirement: Tracked openspec documents are in English
Openspec change documents in the repo SHALL be in English. The drafting and user review stage MAY use Chinese.

#### Scenario: Translate before tracking
- **WHEN** a maintainer commits a Chinese openspec change directory that was not tracked before
- **THEN** the committed content is the English translation, and the commit message is in English
