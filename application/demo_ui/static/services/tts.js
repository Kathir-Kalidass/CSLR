import { now } from "../core/utils.js";

export function createTTSService({ state }) {
  async function speakSentence(text) {
    if (state.speaking) {
      return;
    }
    state.speaking = true;

    try {
      const resp = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang: "en", slow: false }),
      });

      if (!resp.ok) {
        throw new Error(`TTS failed: ${resp.status}`);
      }

      const blob = await resp.blob();
      const audioUrl = URL.createObjectURL(blob);

      if (state.currentAudio) {
        state.currentAudio.pause();
      }

      const audio = new Audio(audioUrl);
      state.currentAudio = audio;
      await audio.play();

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };
    } catch {
      const audioState = document.getElementById("audioState");
      if (audioState) {
        audioState.textContent = "tts_error";
      }
    } finally {
      state.speaking = false;
    }
  }

  async function maybeSpeak(payload) {
    if (!state.ttsEnabled || !state.running) {
      return;
    }

    const sentence = (payload.final_sentence || "").trim();
    if (!sentence || sentence.toLowerCase().includes("press start")) {
      return;
    }

    const elapsed = now() - state.lastSpokenAt;
    if (sentence === state.lastSentenceSpoken || elapsed < 4200) {
      return;
    }

    state.lastSentenceSpoken = sentence;
    state.lastSpokenAt = now();
    await speakSentence(sentence);
  }

  return {
    maybeSpeak,
    speakSentence,
  };
}
