## ADDED Requirements

### Requirement: Commit messages contain no Chinese
The subject and body of a new commit SHALL NOT contain CJK characters or fullwidth punctuation. The local `commit-msg` hook SHALL run the check. History commits SHALL NOT be rewritten.

#### Scenario: A Chinese subject is rejected
- **WHEN** a developer commits with a subject that contains Chinese characters, and the `commit-msg` hook is installed
- **THEN** the hook exits with a non-zero code, and no commit is created

#### Scenario: A body with fullwidth punctuation is rejected
- **WHEN** a developer commits with a body that contains fullwidth punctuation, and the `commit-msg` hook is installed
- **THEN** the hook exits with a non-zero code, and no commit is created

#### Scenario: Non-CJK non-ASCII characters are allowed
- **WHEN** a commit body contains `→` or `é` and no Chinese characters
- **THEN** the check passes

### Requirement: New ignore rules do not hit tracked files by mistake
Each addition to `.gitignore` SHALL ensure that `git ls-files -ci --exclude-standard` shows no new hits (baseline on 2026-10-13: 21).

#### Scenario: Check tracked files after adding a rule
- **WHEN** a maintainer adds a new rule to `.gitignore`
- **THEN** the output of `git ls-files -ci --exclude-standard` is the same as before the rule was added

### Requirement: Openspec documents that enter the repo are in English
Openspec change documents that enter the repo SHALL be in English. The drafting and user review stage MAY use Chinese.

#### Scenario: Translate before entering the repo
- **WHEN** a maintainer commits a Chinese openspec change directory that was not in the repo before
- **THEN** the committed content is the English translation, and the commit message is in English
