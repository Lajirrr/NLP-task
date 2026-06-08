const sourceText = document.querySelector("#sourceText");
const translateButton = document.querySelector("#translateButton");
const statusText = document.querySelector("#statusText");
const translationResult = document.querySelector("#translationResult");

function setResult(text, state) {
  translationResult.textContent = text;
  translationResult.classList.remove("is-muted", "is-error", "is-success");
  if (state) {
    translationResult.classList.add(state);
  }
}

function setLoading(isLoading) {
  translateButton.disabled = isLoading;
  translateButton.textContent = isLoading ? "翻译中" : "翻译";
  statusText.textContent = isLoading ? "模型解码中" : "等待输入";
}

async function translateCurrentText() {
  const text = sourceText.value.trim();
  if (!text) {
    statusText.textContent = "需要输入";
    setResult("请输入一句英文。", "is-error");
    return;
  }

  setLoading(true);
  setResult("正在翻译...", "is-muted");

  try {
    const response = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Translation failed.");
    }
    statusText.textContent = "完成";
    setResult(payload.translation, "is-success");
  } catch (error) {
    statusText.textContent = "出错";
    setResult(error.message || "翻译失败，请稍后再试。", "is-error");
  } finally {
    setLoading(false);
  }
}

translateButton.addEventListener("click", translateCurrentText);
sourceText.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    translateCurrentText();
  }
});
