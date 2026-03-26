export function createPopupController({ popupBackdrop, popupPanel }) {
  function hidePopup() {
    popupBackdrop.classList.add("hidden");
    popupPanel.classList.add("hidden");
  }

  function showPopup(title, contentHtml) {
    popupPanel.innerHTML = `
      <div class="popup-header">
        <h3>${title}</h3>
        <button id="closePopup" class="popup-close">Close</button>
      </div>
      <div class="popup-body">${contentHtml}</div>
    `;

    popupBackdrop.classList.remove("hidden");
    popupPanel.classList.remove("hidden");

    if (window.gsap) {
      window.gsap.fromTo(
        popupPanel,
        { opacity: 0, y: 24 },
        { opacity: 1, y: 0, duration: 0.25, ease: "power2.out" },
      );
    }

    const closeBtn = document.getElementById("closePopup");
    if (closeBtn) {
      closeBtn.addEventListener("click", hidePopup);
    }
  }

  popupBackdrop.addEventListener("click", hidePopup);

  return {
    showPopup,
    hidePopup,
  };
}
