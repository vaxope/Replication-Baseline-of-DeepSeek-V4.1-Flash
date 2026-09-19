#!/bin/bash
mkdir -p /logs/verifier

# Fail safely if answer file does not exist
if [ ! -f /app/answer.txt ]; then
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

# Trim all whitespace, newlines, and carriage returns
actual=$(tr -d '[:space:]' < /app/answer.txt)
expected=$(tr -d '[:space:]' < /tests/expected_answer.txt)

if [ "$actual" = "$expected" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
