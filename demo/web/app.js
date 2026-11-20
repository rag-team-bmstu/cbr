const form = document.querySelector("#search");
const input = document.querySelector("#question");
const submit = document.querySelector("#submit");
const status = document.querySelector("#status");
const answerPanel = document.querySelector("#answer-panel");
const answerText = document.querySelector("#answer");
const modelText = document.querySelector("#model");
const sourcesPanel = document.querySelector("#sources-panel");
const sourcesList = document.querySelector("#sources");

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.question;
    input.focus();
  });
});

function sourceLink(text, url) {
  const link = document.createElement("a");
  link.textContent = text;
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function showSources(sources) {
  sourcesList.replaceChildren(...sources.map((source) => {
    const item = document.createElement("li");
    item.value = source.number;
    const section = document.createElement("p");
    section.className = "metadata";
    section.textContent = source.section;
    if (source.updated) {
      const labels = {
        effective_law: "Редакция действует с",
        archive_publication: "Опубликовано",
      };
      const label = labels[source.kind] || "Обновлено";
      section.append(` · ${label} ${source.updated.split("-").reverse().join(".")}`);
    }
    const text = document.createElement("p");
    text.textContent = source.text;
    item.append(sourceLink(source.title, source.url), section, text);
    return item;
  }));
  sourcesPanel.hidden = sources.length === 0;
}

function showAnswer(data) {
  const urls = new Map(data.sources.map((source) => [String(source.number), source.url]));
  answerText.replaceChildren();
  let position = 0;
  for (const match of data.answer.matchAll(/\[(\d+)\]/g)) {
    answerText.append(data.answer.slice(position, match.index));
    const url = urls.get(match[1]);
    answerText.append(url ? sourceLink(match[0], url) : match[0]);
    position = match.index + match[0].length;
  }
  answerText.append(data.answer.slice(position));
  modelText.textContent = `${data.provider} · ${data.model}`;
  answerPanel.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submit.disabled) return;
  const question = input.value.trim();
  status.dataset.error = "false";
  answerPanel.hidden = true;
  sourcesPanel.hidden = true;
  if (!question) {
    status.textContent = "Введите вопрос.";
    input.focus();
    return;
  }

  status.textContent = "Ищем источники и готовим ответ…";
  submit.disabled = true;
  try {
    const response = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) {
      status.dataset.error = "true";
      status.textContent = data.error;
      return;
    }
    showSources(data.sources);
    if (data.status === "no_matches") {
      status.textContent = "В корпусе не найдено совпадений. Попробуйте уточнить вопрос.";
    } else if (data.status === "generation_error") {
      status.dataset.error = "true";
      status.textContent = data.error;
    } else {
      status.textContent = "";
      showAnswer(data);
    }
  } catch {
    status.dataset.error = "true";
    status.textContent = "Не удалось обработать запрос. Проверьте, что локальный сервер запущен.";
  } finally {
    submit.disabled = false;
  }
});
