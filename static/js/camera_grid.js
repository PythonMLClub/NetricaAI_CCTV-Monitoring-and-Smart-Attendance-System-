document.addEventListener("DOMContentLoaded", () => {
  const cameraModal = document.getElementById("cameraModal");
  const cameraFeedText = document.getElementById("cameraFeedText");
  const cameraModalLabel = document.getElementById("cameraModalLabel");

  cameraModal.addEventListener("show.bs.modal", event => {
    const button = event.relatedTarget;
    const cameraName = button.getAttribute("data-camera");

    cameraModalLabel.textContent = cameraName;
    cameraFeedText.textContent = "Live Feed - " + cameraName;
  });
});