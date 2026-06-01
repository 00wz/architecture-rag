#!/usr/bin/env bash
# test_examples.sh
# ================
# Прогоняет демонстрационные вопросы через REST API RAG-бота (curl) и
# печатает ответы в консоль: 5 успешных диалогов + 2 кейса «Я не знаю».
#
# Перед запуском:
#   1) заданы LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
#   2) поднят сервер:  .venv/bin/uvicorn app:app --port 8000
#
# Запуск:
#   ./test_examples.sh                 # API на http://localhost:8000
#   API_URL=http://host:8000 ./test_examples.sh

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

# 5 вопросов с ответом в базе + 2 без ответа («Я не знаю»)
SUCCESS=(
  "Кто воспитал осиротевшего Korin Valtaar и на какой планете?"
  "Какой дроид никогда не получал полной очистки памяти?"
  "Откуда родом пилот истребителя Joss Varek и в каком флоте он служил сначала?"
  "Кем был Lorin Castrel и как он потерял корабль Nova Drifter?"
  "Who was Tovan Reyes within the Skarn Order?"
)
UNKNOWN=(
  "Какое любимое блюдо у Korin Valtaar?"
  "Сколько детей было у дроида A6-N2 и как их звали?"
)

# pretty-printer: если есть jq — форматируем, иначе сырой JSON
fmt() {
  if command -v jq >/dev/null 2>&1; then
    jq -r '"\nОТВЕТ:\n" + .answer + "\n\nИСТОЧНИКИ: " + ([.sources[] | .character_name + " (" + .source + ")"] | join(", "))'
  else
    cat
  fi
}

ask() {
  local q="$1"
  echo "================================================================"
  echo "ВОПРОС: $q"
  echo "================================================================"
  curl -s -X POST "$API_URL/ask" \
    -H "Content-Type: application/json" \
    -d "$(printf '{"query": %s}' "$(printf '%s' "$q" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
    | fmt
  echo
}

echo "### Проверка здоровья сервиса"
if command -v jq >/dev/null 2>&1; then
  curl -s "$API_URL/health" | jq .
else
  curl -s "$API_URL/health"; echo
fi

echo
echo "##############  УСПЕШНЫЕ ДИАЛОГИ (ответ есть в базе)  ##############"
for q in "${SUCCESS[@]}"; do ask "$q"; done

echo
echo "##############  КЕЙСЫ «Я НЕ ЗНАЮ» (ответа в базе нет)  ##############"
for q in "${UNKNOWN[@]}"; do ask "$q"; done
